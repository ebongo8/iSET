import torch
import pytest
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency, LayerGradCam
from test_files.utils import (
    get_device_type,
    load_trained_model,
    get_mnist_image,
    adjust_brightness,
    blur_image,
    create_saliency_and_gradcam_heatmaps,
    remove_non_salient_pixels,
    apply_gaussian_noise
)


def get_trained_model_for_xai_tests():
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")
    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()
    return model, device


def compute_saliency(model, device, img_array, target_class):
    """
    Returns a normalized saliency map (numpy 28x28)
    """
    img_tensor = torch.tensor(img_array, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    saliency = Saliency(model)
    attr = saliency.attribute(img_tensor, target=target_class)
    heatmap = attr.squeeze().detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# --------------------------------
# TC-XAI-01: Shape Sensitivity
# --------------------------------
def test_tc_xai_01_shape_sensitivity():
    """
    Corresponds to TC-CF-01 Shape sensitivity - saliency.
     Visualize which pixels drive decision changes when loops/lines are perturbed. Determine
     whether the model’s attention regions (saliency maps) align with the pixels
      modified during geometric edits.
    Expected result: Saliency maps highlight modified stroke regions;
    Causal focus aligns with the manipulated feature.
    """
    model, device = get_trained_model_for_xai_tests()
    # Use same images as TC-CF-01
    ORIG_DIR = "../test_images/CF_images/TC-CF-01_original"
    MOD_DIR = "../test_images/CF_images/TC-CF-01_modified"

    orig_files = sorted(os.listdir(ORIG_DIR))
    mod_files = sorted(os.listdir(MOD_DIR))
    results = []

    for orig_fname in orig_files:
        prefix_match = re.match(r"^(Q\d+_)", orig_fname)
        if not prefix_match:
            continue
        prefix = prefix_match.group(1)

        mod_fname = next((f for f in mod_files if f.startswith(prefix)), None)
        if not mod_fname:
            continue

        orig_img = plt.imread(os.path.join(ORIG_DIR, orig_fname))
        mod_img = plt.imread(os.path.join(MOD_DIR, mod_fname))

        digit_match = re.search(r"mnist_(\d)", orig_fname)
        true_label = int(digit_match.group(1))

        img_tensor_orig = torch.tensor(orig_img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor_mod = torch.tensor(mod_img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        pred_mod = torch.argmax(model(img_tensor_mod)).item()

        # ---- Use helper for both saliency + GradCAM ----
        sal_o, gc_o = create_saliency_and_gradcam_heatmaps(model, img_tensor_orig, true_label)
        sal_m, gc_m = create_saliency_and_gradcam_heatmaps(model, img_tensor_mod, true_label)

        # ---- Visualization ----
        fig, axes = plt.subplots(2, 3, figsize=(10, 5))

        # ----- ORIGINAL ROW -----
        axes[0, 0].imshow(orig_img, cmap='gray')
        axes[0, 0].set_title(f"Original (Label: {true_label})")

        axes[0, 1].imshow(sal_o, cmap='hot')
        axes[0, 1].set_title("Saliency Orig")

        axes[0, 2].imshow(gc_o, cmap='hot')
        axes[0, 2].set_title("GradCAM Orig")

        # ----- MODIFIED ROW -----
        axes[1, 0].imshow(mod_img, cmap='gray')
        axes[1, 0].set_title(f"Modified (Pred: {pred_mod})")

        axes[1, 1].imshow(sal_m, cmap='hot')
        axes[1, 1].set_title("Saliency Mod")

        axes[1, 2].imshow(gc_m, cmap='hot')
        axes[1, 2].set_title("GradCAM Mod")

        plt.tight_layout()
        plt.savefig(f"TC-XAI-01_{prefix}_heatmaps.png")
        plt.close()

        results.append(pred_mod == true_label)
    # Don't need to assert anything for this test, just print out heatmaps
    # assert all(results), "Some shape-perturbed images caused misclassification"


# --------------------------------
# TC-XAI-02: Blur
# --------------------------------
def test_tc_xai_02_blur():
    """
    Corresponds to TC-CF-03 blur - saliency
    Evaluate attention redistribution under blurred input. Assess if blurred regions lose saliency
    weight proportionally while model focus remains on unblurred semantic zones.

    Expected result: SSaliency maps should maintain focus on digit body; decreased intensity at blurred
    regions without spurious hotspots.
    """
    model, device = get_trained_model_for_xai_tests()

    orig_img, label = get_mnist_image(target_digit=8, target_index=0, show=False)

    # Create a mask for the top loop
    # Approx region: upper half (rows 0–13)
    mask = np.zeros_like(orig_img)
    mask[0:14, :] = 1.0

    # Apply Gaussian blur
    blurred_full = blur_image(orig_img, sigma=1.0)

    # Blend so only the masked region is blurred
    perturbed_img = orig_img * (1 - mask) + blurred_full * mask

    img_t_orig = torch.tensor(orig_img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_t_blur = torch.tensor(perturbed_img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sal_o, gc_o = create_saliency_and_gradcam_heatmaps(model, img_t_orig, label)
    sal_b, gc_b = create_saliency_and_gradcam_heatmaps(model, img_t_blur, label)

    pred_orig = torch.argmax(model(img_t_orig)).item()
    pred_blur = torch.argmax(model(img_t_blur)).item()

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes[0, 0].imshow(orig_img, cmap='gray')
    axes[0, 0].set_title(f"Original (Label: {label})")

    axes[0, 1].imshow(sal_o, cmap='hot')
    axes[0, 1].set_title("Saliency Orig")
    axes[0, 2].imshow(gc_o, cmap='hot')
    axes[0, 2].set_title("GradCAM Orig")

    axes[1, 0].imshow(perturbed_img, cmap='gray')
    axes[1, 0].set_title(f"Blurred (Pred: {pred_blur})")

    axes[1, 1].imshow(sal_b, cmap='hot')
    axes[1, 1].set_title("Saliency Blur")
    axes[1, 2].imshow(gc_b, cmap='hot');
    axes[1, 2].set_title("GradCAM Blur")

    plt.tight_layout()
    plt.savefig("TC-XAI-02_blur_heatmaps.png")
    plt.close()

    # assert pred_orig == pred_blur, "Blur caused misclassification"


# --------------------------------
# TC-XAI-03: Noise + Pixel Removal (Saliency)
# --------------------------------
def test_tc_xai_03_noise():
    """
    XAI counterpart to TC-CF-04:
    Generates two heatmap figures per digit:
        1) Noise:   Original / 10% Noise / 40% Noise  (3x3)
        2) Removal: Original / Remove 5px / Remove 150px (3x3)

    Each row shows: Image | Saliency | GradCAM
    """

    model, device = get_trained_model_for_xai_tests()

    for digit in range(10):

        # -----------------------------
        # Load base MNIST image
        # -----------------------------
        orig_img, label = get_mnist_image(
            target_digit=digit, target_index=0, show=False
        )
        assert label == digit

        # Convert image to torch
        img_t_o = torch.tensor(orig_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # ----------------------------------------------------------
        # ORIGINAL saliency + GradCAM
        # ----------------------------------------------------------
        sal_o, gc_o = create_saliency_and_gradcam_heatmaps(model, img_t_o, digit)

        # ----------------------------------------------------------
        # GENERATE NOISY IMAGES (10% and 40%)
        # ----------------------------------------------------------
        noisy_10 = apply_gaussian_noise(orig_img, amount=0.10)
        noisy_40 = apply_gaussian_noise(orig_img, amount=0.40)

        img_t_n10 = torch.tensor(noisy_10, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        img_t_n40 = torch.tensor(noisy_40, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        sal_n10, gc_n10 = create_saliency_and_gradcam_heatmaps(model, img_t_n10, digit)
        sal_n40, gc_n40 = create_saliency_and_gradcam_heatmaps(model, img_t_n40, digit)

        # ----------------------------------------------------------
        # FIND LAST CONV LAYER FOR GRADCAM (for removal saliency)
        # ----------------------------------------------------------
        last_conv_layer = None
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                last_conv_layer = layer
        assert last_conv_layer is not None

        gradcam = LayerGradCam(model, last_conv_layer)

        # ----------------------------------------------------------
        # Compute GradCAM heatmap for pixel ranking
        # ----------------------------------------------------------
        with torch.no_grad():
            out = model(img_t_o)
        pred_orig = out.argmax(1).item()

        cam_attr = gradcam.attribute(img_t_o, target=pred_orig)
        cam_attr = torch.nn.functional.interpolate(
            cam_attr, size=(28, 28), mode="bilinear", align_corners=False
        )
        heatmap = cam_attr.squeeze().detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_t = torch.tensor(heatmap, dtype=torch.float32)

        # ----------------------------------------------------------
        # PIXEL REMOVAL: remove 5 and 150 least salient pixels
        # ----------------------------------------------------------
        removed_5 = remove_non_salient_pixels(orig_img, heatmap_t, 5)
        removed_150 = remove_non_salient_pixels(orig_img, heatmap_t, 150)

        img_t_r5 = torch.tensor(removed_5, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        img_t_r150 = torch.tensor(removed_150, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        sal_r5, gc_r5 = create_saliency_and_gradcam_heatmaps(model, img_t_r5, digit)
        sal_r150, gc_r150 = create_saliency_and_gradcam_heatmaps(model, img_t_r150, digit)

        # ==================================================================
        #  FIGURE 1 — NOISE (3×3)
        # ==================================================================
        fig_n, ax = plt.subplots(3, 3, figsize=(9, 9))

        # Row 1 — ORIGINAL
        ax[0, 0].imshow(orig_img, cmap='gray')
        ax[0, 0].set_title("Original")
        ax[0, 1].imshow(sal_o, cmap='hot')
        ax[0, 1].set_title("Saliency Orig")
        ax[0, 2].imshow(gc_o, cmap='hot')
        ax[0, 2].set_title("GradCAM Orig")

        # Row 2 — 10% NOISE
        ax[1, 0].imshow(noisy_10, cmap='gray')
        ax[1, 0].set_title("10% Noise")
        ax[1, 1].imshow(sal_n10, cmap='hot')
        ax[1, 1].set_title("Saliency 10%")
        ax[1, 2].imshow(gc_n10, cmap='hot')
        ax[1, 2].set_title("GradCAM 10%")

        # Row 3 — 40% NOISE
        ax[2, 0].imshow(noisy_40, cmap='gray')
        ax[2, 0].set_title("40% Noise")
        ax[2, 1].imshow(sal_n40, cmap='hot')
        ax[2, 1].set_title("Saliency 40%")
        ax[2, 2].imshow(gc_n40, cmap='hot')
        ax[2, 2].set_title("GradCAM 40%")

        for r in range(3):
            for c in range(3):
                ax[r, c].axis("off")

        fig_n.suptitle(f"TC-XAI-03 — Noise Heatmaps (Digit {digit})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_n.savefig(f"TC-XAI-03_digit_{digit}_noise_heatmaps.png")
        plt.close(fig_n)

        # ==================================================================
        #  FIGURE 2 — PIXEL REMOVAL (3×3)
        # ==================================================================
        fig_p, axp = plt.subplots(3, 3, figsize=(9, 9))

        # Row 1 — ORIGINAL
        axp[0, 0].imshow(orig_img, cmap='gray')
        axp[0, 0].set_title("Original")
        axp[0, 1].imshow(sal_o, cmap='hot')
        axp[0, 1].set_title("Saliency Orig")
        axp[0, 2].imshow(gc_o, cmap='hot')
        axp[0, 2].set_title("GradCAM Orig")

        # Row 2 — Remove 5
        axp[1, 0].imshow(removed_5, cmap='gray')
        axp[1, 0].set_title("Remove 5px")
        axp[1, 1].imshow(sal_r5, cmap='hot')
        axp[1, 1].set_title("Saliency 5px")
        axp[1, 2].imshow(gc_r5, cmap='hot')
        axp[1, 2].set_title("GradCAM 5px")

        # Row 3 — Remove 150
        axp[2, 0].imshow(removed_150, cmap='gray')
        axp[2, 0].set_title("Remove 150px")
        axp[2, 1].imshow(sal_r150, cmap='hot')
        axp[2, 1].set_title("Saliency 150px")
        axp[2, 2].imshow(gc_r150, cmap='hot')
        axp[2, 2].set_title("GradCAM 150px")

        for r in range(3):
            for c in range(3):
                axp[r, c].axis("off")

        fig_p.suptitle(f"TC-XAI-03 — Pixel Removal Heatmaps (Digit {digit})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_p.savefig(f"TC-XAI-03_digit_{digit}_pixelremoval_heatmaps.png")
        plt.close(fig_p)


'''
# ---------------------------------------
# FUTURE WORK XAI TEST Brighten & Strokes
# ---------------------------------------
def test_tc_xai_02_brightness_strokes():
    """
    Corresponds to TC-CF-03 brighten and strokes - saliency
     Analyze if saliency redistributes under lighting or stroke changes.

    Expected result: Saliency should remain centered on digit-defining regions;
     Minimal shift under brightness variation.
    """
    model, device = get_trained_model_for_xai_tests()

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        dark_img = adjust_brightness(orig_img, factor=0.9)
        img_t_orig = torch.tensor(orig_img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_t_dark = torch.tensor(dark_img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sal_o, gc_o = create_saliency_and_gradcam_heatmaps(model, img_t_orig, digit)
        sal_d, gc_d = create_saliency_and_gradcam_heatmaps(model, img_t_dark, digit)
        
        fig, axes = plt.subplots(2, 3, figsize=(9, 5))
        axes[0, 0].imshow(orig_img, cmap='gray')
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(sal_o, cmap='hot')
        axes[0, 1].set_title("Saliency Orig")
        axes[0, 2].imshow(gc_o, cmap='hot')
        axes[0, 2].set_title("GradCAM Orig")

        axes[1, 0].imshow(dark_img, cmap='gray')
        axes[1, 0].set_title("Darkened")
        axes[1, 1].imshow(sal_d, cmap='hot')
        axes[1, 1].set_title("Saliency Dark")
        axes[1, 2].imshow(gc_d, cmap='hot')
        axes[1, 2].set_title("GradCAM Dark")

        plt.tight_layout()
        plt.savefig(f"TC-XAI-02_digit_{digit}_heatmaps.png")
        plt.close()
'''
