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
    flip_image,
    adjust_brightness,
    add_noise,
    blur_image,
    compute_flip_rate,
    create_saliency_and_gradcam_heatmaps
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
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))

        # ----- ORIGINAL ROW -----
        axes[0, 0].imshow(orig_img, cmap='gray')
        axes[0, 0].set_title(f"Original (Label: {true_label})")

        axes[0, 1].imshow(sal_o, cmap='hot')
        axes[0, 1].set_title("Saliency Orig")

        axes[0, 2].imshow(gc_o, cmap='hot')
        axes[0, 2].set_title("GradCAM Orig")

        axes[0, 3].axis('off')

        # ----- MODIFIED ROW -----
        axes[1, 0].imshow(mod_img, cmap='gray')
        axes[1, 0].set_title(f"Modified (Pred: {pred_mod})")

        axes[1, 1].imshow(sal_m, cmap='hot')
        axes[1, 1].set_title("Saliency Mod")

        axes[1, 2].imshow(gc_m, cmap='hot')
        axes[1, 2].set_title("GradCAM Mod")

        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"TC-XAI-01_{prefix}_heatmaps.png")
        plt.close()

        results.append(pred_mod == true_label)
    # Don't need to assert anything for this test, just print out heatmaps
    # assert all(results), "Some shape-perturbed images caused misclassification"


# --------------------------------
# TC-XAI-02: Brighten & Strokes
# --------------------------------
def test_tc_xai_02_brightness_strokes():
    """
    Corresponds to TC-CF-03 brighten and strokes - saliency
     Analyze if saliency redistributes under lighting or stroke changes.

    Expected result: Saliency should remain centered on digit-defining regions;
     Minimal shift under brightness variation.
    """
    model, device = get_trained_model_for_xai_tests()
    before_imgs, after_imgs = [], []
    heatmaps_before, heatmaps_after = [], []

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        dark_img = adjust_brightness(orig_img, factor=0.9)
        before_imgs.append(orig_img)
        after_imgs.append(dark_img)

        heatmaps_before.append(compute_saliency(model, device, orig_img, digit))
        heatmaps_after.append(compute_saliency(model, device, dark_img, digit))

    # Save example saliency maps
    for i in range(10):
        fig, axes = plt.subplots(1, 2, figsize=(5,3))
        axes[0].imshow(before_imgs[i], cmap='gray'); axes[0].imshow(heatmaps_before[i], cmap='hot', alpha=0.5)
        axes[0].set_title(f'Digit {i} Original')
        axes[0].axis('off')
        axes[1].imshow(after_imgs[i], cmap='gray'); axes[1].imshow(heatmaps_after[i], cmap='hot', alpha=0.5)
        axes[1].set_title(f'Digit {i} Darkened')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(f"TC-XAI-02_digit_{i}_saliency.png")
        plt.close(fig)


# --------------------------------
# TC-XAI-03: Blur
# --------------------------------
def test_tc_xai_03_blur():
    """
    Corresponds to TC-CF-04 blur - saliency
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
    plt.savefig("TC-XAI-03_blur_heatmaps.png")
    plt.close()

    # assert pred_orig == pred_blur, "Blur caused misclassification"


# --------------------------------
# TC-XAI-04: Noise
# --------------------------------
def test_tc_xai_04_noise():
    model, device = get_trained_model_for_xai_tests()
    originals, noisy_imgs, heatmaps_orig, heatmaps_noisy = [], [], [], []

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        noisy_img = add_noise(orig_img, amount=0.1)
        originals.append(orig_img)
        noisy_imgs.append(noisy_img)

        heatmaps_orig.append(compute_saliency(model, device, orig_img, digit))
        heatmaps_noisy.append(compute_saliency(model, device, noisy_img, digit))

    # Save visualization
    for i in range(10):
        fig, axes = plt.subplots(1,2,figsize=(5,3))
        axes[0].imshow(originals[i], cmap='gray'); axes[0].imshow(heatmaps_orig[i], cmap='hot', alpha=0.5)
        axes[0].set_title(f'Digit {i} Original')
        axes[0].axis('off')
        axes[1].imshow(noisy_imgs[i], cmap='gray'); axes[1].imshow(heatmaps_noisy[i], cmap='hot', alpha=0.5)
        axes[1].set_title(f'Digit {i} Noisy')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(f"TC-XAI-04_digit_{i}_noise_saliency.png")
        plt.close(fig)
