import torch
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import re  # regular expressions package
from PIL import Image
from test_files.utils import (
    get_device_type,
    load_trained_model,
    get_mnist_image,
    flip_image,
    adjust_brightness,
    add_noise,
    blur_image,
    compute_proximity_delta,
    compute_flip_rate
)
from captum.attr import Saliency, LayerGradCam


# --------------------------------
# Helper Functions for CF Tests
# --------------------------------
def get_trained_model_for_cf_tests():
    """
    Load the trained model and return it in evaluation mode on the correct device.
    """
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)

    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")

    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()
    return model, device


def predict_class(model, device, img_array):
    """
    Converts a numpy image to tensor, forwards through model, returns predicted class.
    """
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    return pred_class


def load_image(path):
    """Loads original image (no normalization required) for visualization + proximity delta."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def find_matching_modified(prefix, mod_files):
    """Return the first modified file with matching prefix."""
    pattern = rf"^{re.escape(prefix)}"
    for fname in mod_files:
        if re.match(pattern, fname):
            return fname
    return None


def test_tc_cf_01_shape_sensitivity():
    """
    TC-CF-01 Shape Sensitivity: Evaluate model robustness to small geometric changes
    such as closing/opening loops, extending tails, merging lines. Counterfactual metrics
    outputted are flip rate and proximity delta for analysis.

    Expected result: Minimal change in classification; small shape perturbations should
    not trigger label flips unless they cross semantic boundaries. Proximity delta values
    within defined threshold.
    """
    threshold = 0.03
    ORIG_DIR = "../test_images/CF_images/TC-CF-01_original"
    MOD_DIR = "../test_images/CF_images/TC-CF-01_modified"

    model, device = get_trained_model_for_cf_tests()

    orig_files = sorted(os.listdir(ORIG_DIR))
    mod_files = sorted(os.listdir(MOD_DIR))

    results = []

    for orig_fname in orig_files:
        # Extract shared prefix ("Q1_", "Q2_", ...)
        prefix_match = re.match(r"^(Q\d+_)", orig_fname)
        if not prefix_match:
            continue

        prefix = prefix_match.group(1)
        mod_fname = find_matching_modified(prefix, mod_files)
        if not mod_fname:
            raise FileNotFoundError(f"No modified file found for prefix {prefix}")

        orig_path = os.path.join(ORIG_DIR, orig_fname)
        mod_path = os.path.join(MOD_DIR, mod_fname)

        # Load images
        orig_img_arr = load_image(orig_path)   # numpy float32, 0–1
        mod_img_arr = load_image(mod_path)     # numpy float32, 0–1

        # Infer true digit from filename
        digit_match = re.search(r"mnist_(\d)", orig_fname)
        if not digit_match:
            raise ValueError(f"Cannot infer digit label from filename: {orig_fname}")
        true_label = int(digit_match.group(1))

        # Predict
        pred_label = predict_class(model, device, mod_img_arr)
        # Proximity delta (dict with L1 and L2)
        prox = compute_proximity_delta(orig_img_arr, mod_img_arr)

        results.append({
            "prefix": prefix,
            "orig_fname": orig_fname,
            "mod_fname": mod_fname,
            "orig_img": orig_img_arr,
            "mod_img": mod_img_arr,
            "true": true_label,
            "pred": pred_label,
            "prox": prox,
            "pass_pred": (pred_label == true_label),
            "pass_prox": (prox["L1"] < threshold)  # threshold on L1
        })

    # ---------------------------------------------------------
    #   CREATE SUMMARY PNG + FLIP RATE
    # ---------------------------------------------------------
    # SORT RESULTS IN ASCENDING Q-ORDER
    results.sort(key=lambda r: int(re.search(r"Q(\d+)_", r["prefix"]).group(1)))

    # Compute flip rate for the figure title
    flip_rate = compute_flip_rate([
        (None, None, res["true"], None, res["pred"])
        for res in results
    ])

    # Count how many did not meet the proximity threshold
    num_fail_prox = sum(1 for r in results if not r["pass_prox"])

    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 3 * n))

    fig.suptitle(
        f"TC-CF-01 Shape Sensitivity — Original vs Modified\n"
        f"Flip Rate = {flip_rate:.2f}%\n"
        f"{num_fail_prox}/{n} images > proximity delta threshold ({threshold})",
        fontsize=24,
        y=0.98
    )

    if n == 1:
        axes = np.array([axes])  # Keep 2D indexing consistent

    for i, res in enumerate(results):
        ax_orig = axes[i, 0]
        ax_mod = axes[i, 1]

        # Original image
        ax_orig.imshow(res["orig_img"], cmap="gray")
        ax_orig.set_title(
            f"Original\nDigit {res['true']}",
            fontsize=18
        )
        ax_orig.axis("off")

        # Modified image + metrics
        status_pred = "PASS" if res["pass_pred"] else "FAIL"
        status_prox = "PASS" if res["pass_prox"] else "FAIL"

        ax_mod.imshow(res["mod_img"], cmap="gray")
        ax_mod.set_title(
            f"Predicted: {res['pred']} ({status_pred})\n"
            # f"L1 Δ={res['prox']['L1']:.4f}, L2 Δ={res['prox']['L2']:.4f} ({status_prox})"
            f"L1 Δ={res['prox']['L1']:.4f} ({status_prox})",
            fontsize=18
        )
        ax_mod.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("TC-CF-01_shape_sensitivity_summary.png")
    plt.close()

    failures = [
        res for res in results
        if not res["pass_pred"] or not res["pass_prox"]
    ]
    # Remove img arrays from res dictionaries so they don't print if assertion fails
    for res in failures:
        res.pop("orig_img", None)
        res.pop("mod_img", None)

    assert not failures, (
        "TC-CF-01 failed for:\n" +
        "\n".join(
            f"{res['prefix']} true={res['true']} pred={res['pred']} "
            f"L1={res['prox']['L1']:.4f}, L2={res['prox']['L2']:.4f}"
            for res in failures
        )
    )


# ----------------------------
# TC-CF-02: Flip image
# ----------------------------
def test_tc_cf_02_flip_image():
    """
    TC-CF-01 Flip Image: Measure spatial invariance by horizontally mirroring input digits.
    Counterfactual metric outputted is stability for analysis.
    Expected result: Classification should remain stable after image flipping (same class prediction).
    """
    model, device = get_trained_model_for_cf_tests()
    # change this to the indices you want to test (e.g. range(5), range(100), etc.)
    indices = range(5)

    results = []  # (index, orig_img, true_label, flipped_img, pred_label)

    # Process all indices, collect results (do not assert inside the loop)
    for i in indices:
        orig_image, label = get_mnist_image(index=i, train_data_set=True, show=False)
        flipped_image = flip_image(orig_image)
        pred_label = predict_class(model, device, flipped_image)
        results.append((i, orig_image, label, flipped_image, pred_label))

    # Compute flip rate from results
    flip_rate = compute_flip_rate(results)

    # ---------------------------------------------------------------------
    # Create the PNG summary for all processed indices
    # ---------------------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(6, 2 * n))
    fig.suptitle(
        f"TC-CF-02 Flip Image Results\nFlip Rate: {flip_rate:.1f}%",
        fontsize=16
    )

    for row, (idx, orig_img, true_label, flip_img, pred_label) in enumerate(results):
        ax_orig = axes[row, 0] if n > 1 else axes[0]
        ax_flip = axes[row, 1] if n > 1 else axes[1]

        ax_orig.imshow(orig_img, cmap="gray")
        ax_orig.set_title(f"Original (Label: {true_label})")
        ax_orig.axis("off")

        ax_flip.imshow(flip_img, cmap="gray")
        ax_flip.set_title(f"Flipped (Predicted: {pred_label})")
        ax_flip.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("TC-CF-02_flip_image_results.png")
    plt.close()

    # ---------------------------------------------------------------------
    # Single assertion at the end: fail if any prediction changed
    # ---------------------------------------------------------------------
    mismatches = [
        (idx, true_label, pred_label)
        for idx, _, true_label, _, pred_label in results
        if pred_label != true_label
    ]

    assert not mismatches, (
        f"Prediction changed after horizontal flip for {len(mismatches)} samples. "
        f"Examples (index, true, pred): {mismatches[:10]}"
    )


# ----------------------------
# TC-CF-03 Helper Function
# ----------------------------
def save_visualization(before_img, after_img, filename, pred_before=None, pred_after=None, title=None, multiple=False,
                       before_imgs=None, after_imgs=None, titles=None):
    """
    TC-CF-03 helper to save before/after comparisons.

    Save MNIST visualizations.

    If multiple=True, before_imgs and after_imgs should be lists of images (for Case 2 & 3).
    Otherwise, before_img and after_img are single images (for Case 1).
    """
    if multiple:
        n_rows = len(before_imgs)
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))

        if n_rows == 1:
            axes = np.array([axes])  # make iterable
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_rows):
            axes[i, 0].imshow(before_imgs[i], cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title('Before' if titles is None else titles[i][0], fontsize=8)

            axes[i, 1].imshow(after_imgs[i], cmap='gray')
            axes[i, 1].axis('off')
            axes[i, 1].set_title('After' if titles is None else titles[i][1], fontsize=8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
        axes[0].imshow(before_img, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Before' if pred_before is None else f"{pred_before} (Before)")

        axes[1].imshow(after_img, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('After' if pred_after is None else f"{pred_after} (After)")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ----------------------------
# TC-CF-03: Brighten & Strokes
# ----------------------------
def test_tc_cf_03_brightness_strokes():
    model, device = get_trained_model_for_cf_tests()

    # ============================================================
    # (1) STROKE THICKENING — CASE 1
    # ============================================================
    cf_test_images_dir = "../test_images/CF_images"
    orig_5_png = os.path.join(cf_test_images_dir, "mnist_5_24.png")
    thickened_5_png = os.path.join(cf_test_images_dir, "thicken_stroke_5.png")

    orig_5_arr = load_image(orig_5_png)
    thickened_5_arr = load_image(thickened_5_png)

    pred_orig_5 = predict_class(model, device, orig_5_arr)
    pred_thick_5 = predict_class(model, device, thickened_5_arr)

    save_visualization(
        before_img=orig_5_arr,
        after_img=thickened_5_arr,
        pred_before=pred_orig_5,
        pred_after=pred_thick_5,
        filename="TC-CF-03_case1_stroke_thickening.png",
        title="TC-CF-03 Case 1: Stroke Thickening",
        multiple=False
    )

    results = [(0, orig_5_png, 5, thickened_5_arr, pred_thick_5)]
    flip_rate_strokes = compute_flip_rate(results)
    assert flip_rate_strokes == 0, (
        f"Stroke thickening changed prediction! Flip Rate = {flip_rate_strokes:.1f}%"
    )

    # ============================================================
    # (2) DARKEN STROKES BY 10% — CASE 2
    # ============================================================
    before_imgs, after_imgs, titles_list = [], [], []
    flip_results_darken = []

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        dark_img = adjust_brightness(orig_img, factor=0.9)

        pred_orig = predict_class(model, device, orig_img)
        pred_dark = predict_class(model, device, dark_img)

        before_imgs.append(orig_img)
        after_imgs.append(dark_img)
        titles_list.append([f"Digit {digit} Original", f"Digit {digit} Darkened"])

        flip_results_darken.append((digit, None, digit, dark_img, pred_dark))

    save_visualization(
        before_img=None,
        after_img=None,
        filename="TC-CF-03_case2_darken.png",
        multiple=True,
        before_imgs=before_imgs,
        after_imgs=after_imgs,
        titles=titles_list
    )

    flip_rate_darken = compute_flip_rate(flip_results_darken)
    assert flip_rate_darken < 5, (
        f"[Darken 10%] Flip rate too high: {flip_rate_darken:.2f}%"
    )

    # ============================================================
    # (3) BRIGHTEN BACKGROUND — CASE 3
    # ============================================================
    before_imgs, after_imgs, titles_list = [], [], []
    flip_results_bright = []

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        bright_img = adjust_brightness(orig_img, factor=1.1)

        pred_orig = predict_class(model, device, orig_img)
        pred_bright = predict_class(model, device, bright_img)

        before_imgs.append(orig_img)
        after_imgs.append(bright_img)
        titles_list.append([f"Digit {digit} Original", f"Digit {digit} Brightened"])

        flip_results_bright.append((digit, None, digit, bright_img, pred_bright))

    save_visualization(
        before_img=None,
        after_img=None,
        filename="TC-CF-03_case3_brighten.png",
        multiple=True,
        before_imgs=before_imgs,
        after_imgs=after_imgs,
        titles=titles_list
    )

    flip_rate_bright = compute_flip_rate(flip_results_bright)
    assert flip_rate_bright < 5, (
        f"[Brighten BG] Flip rate too high: {flip_rate_bright:.2f}%"
    )

    # ============================================================
    # SUMMARY PRINT
    # ============================================================
    print("\n=== TC-CF-03 Summary ===")
    print(f"Flip Rate (Stroke Thickening): {flip_rate_strokes:.2f}%")
    print(f"Flip Rate (Darken 10%): {flip_rate_darken:.2f}%")
    print(f"Flip Rate (Brighten BG): {flip_rate_bright:.2f}%")


# ----------------------------
# TC-CF-04: Blur
# ----------------------------
def test_tc_cf_04_blur():
    """
    TC-CF-04 Blur Stability:
    Apply mild Gaussian blur only to the upper loop of digit '8' to test
    localized perceptual degradation. Verify prediction stability and save
    a visualization of before/after images with predicted labels.
    """
    model, device = get_trained_model_for_cf_tests()

    # Load MNIST sample of "8"
    orig_img, label = get_mnist_image(target_digit=8, target_index=0, show=False)
    assert label == 8, "Loaded MNIST image is not the digit '8'"

    # Create a mask for the top loop
    # Approx region: upper half (rows 0–13)
    mask = np.zeros_like(orig_img)
    mask[0:14, :] = 1.0

    # Apply Gaussian blur
    blurred_full = blur_image(orig_img, sigma=1.0)

    # Blend so only the masked region is blurred
    perturbed_img = orig_img * (1 - mask) + blurred_full * mask

    # Predictions
    pred_before = predict_class(model, device, orig_img)
    pred_after = predict_class(model, device, perturbed_img)

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(orig_img, cmap="gray")
    ax[0].set_title(f"Original (pred={pred_before})")
    ax[0].axis("off")

    ax[1].imshow(perturbed_img, cmap="gray")
    ax[1].set_title(f"Blurred Top Loop (pred={pred_after})")
    ax[1].axis("off")

    plt.tight_layout()
    fig.savefig("TC-CF-04_blur_before_after.png")
    plt.close(fig)

    # Assertion: prediction should remain stable
    assert pred_before == pred_after, (
        f"Localized blur caused prediction flip: "
        f"before={pred_before}, after={pred_after}"
    )


# ----------------------------
# TC-CF-05: Helper Functions
# ----------------------------
def get_random_non_salient_pixels(heatmap, num_pixels=5):
    """
    TC-CF-05 Helper:s Return indices of 'num_pixels' random non-salient pixels
    based on the heatmap (low-value pixels = non-salient).
    """
    seed = 100
    torch.manual_seed(seed)  # Set seed for reproducibility

    heatmap = heatmap.clone()

    # Flatten heatmap
    flat = heatmap.flatten()

    # Sort ascending → lowest values = non-salient
    sorted_indices = torch.argsort(flat, descending=False)

    # Choose bottom 80% as non-salient
    k_non_salient = int(0.8 * flat.numel())
    non_salient_indices = sorted_indices[:k_non_salient]

    # Randomly pick N from that pool
    chosen = non_salient_indices[torch.randperm(k_non_salient)[:num_pixels]]

    return chosen


def remove_non_salient_pixels(orig_img, heatmap_t, n_pixels):
    """
    TC-CF-05 Helper: remove N non-salient pixels based on Grad-CAM heatmap
    Remove N lowest-saliency pixels (set to 0).
    Returns modified image.
    """
    idxs = get_random_non_salient_pixels(heatmap_t, num_pixels=n_pixels)

    img_flat = orig_img.copy().flatten()
    img_flat[idxs.numpy()] = 0.0

    return img_flat.reshape(orig_img.shape)


def apply_gaussian_noise(orig_img, amount):
    """
    TC-CF-05 Helper: apply Gaussian noise
    Adds Gaussian noise scaled by `amount` (e.g., 0.10 for 10%).
    """
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, amount, orig_img.shape)
    noisy = orig_img + noise
    return np.clip(noisy, 0, 1)


# ----------------------------
# TC-CF-05: Noise
# ----------------------------
def test_tc_cf_05_noise():
    """
    TC-CF-05 Noise & Pixel Removal Robustness:
    For digits 0–9:
        • Remove 5 low-saliency pixels
        • Remove 150 low-saliency pixels
        • Add 10% Gaussian noise
        • Add 40% Gaussian noise
    Produces two output figures:
        1) Pixel_removal.png
        2) Noise_results.png
    """
    model, device = get_trained_model_for_cf_tests()

    remove_5_results = []
    remove_150_results = []
    noise_10_results = []
    noise_40_results = []

    originals = []
    removed_5_imgs = []
    removed_150_imgs = []
    noisy_10_imgs = []
    noisy_40_imgs = []

    preds_removed_5 = []
    preds_removed_150 = []
    preds_noisy_10 = []
    preds_noisy_40 = []

    # -----------------------------------------------------
    # Find last Conv2D for Grad-CAM
    # -----------------------------------------------------
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = layer
    assert last_conv_layer is not None, "No Conv2D layer found for Grad-CAM."

    gradcam = LayerGradCam(model, last_conv_layer)

    # -----------------------------------------------------
    # Loop through all digits 0–9
    # -----------------------------------------------------
    for digit in range(10):
        orig_img, label = get_mnist_image(
            target_digit=digit, target_index=0, show=False
        )
        assert label == digit
        originals.append(orig_img)

        # -----------------------------------------------------
        # Convert image for Grad-CAM: shape (1,1,28,28)
        # -----------------------------------------------------
        img_tensor = torch.tensor(orig_img, dtype=torch.float32, device=device)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
        pred_class = outputs.argmax(dim=1).item()

        # Compute Grad-CAM heatmap (28×28)
        attr = gradcam.attribute(img_tensor, target=pred_class)

        # Interpolate CAM to 28x28 if needed
        if attr.dim() == 4:
            attr = torch.nn.functional.interpolate(
                attr, size=(28, 28), mode="bilinear", align_corners=False
            )
        else:
            raise ValueError(f"Grad-CAM unexpected shape: {attr.shape}")

        heatmap = attr.squeeze().detach().cpu().numpy()
        # Normalize to [0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # Convert to torch for indexing
        heatmap_t = torch.tensor(heatmap, dtype=torch.float32)

        # =================================================
        # REMOVE 5 PIXELS
        # =================================================
        img_removed_5 = remove_non_salient_pixels(orig_img, heatmap_t, 5)
        pred_removed_5 = predict_class(model, device, img_removed_5)

        removed_5_imgs.append(img_removed_5)
        preds_removed_5.append(pred_removed_5)

        remove_5_results.append((digit, orig_img, label, img_removed_5, pred_removed_5))

        # =================================================
        # REMOVE 150 PIXELS
        # =================================================
        img_removed_150 = remove_non_salient_pixels(orig_img, heatmap_t, 150)
        pred_removed_150 = predict_class(model, device, img_removed_150)

        removed_150_imgs.append(img_removed_150)
        preds_removed_150.append(pred_removed_150)

        remove_150_results.append(
            (digit, orig_img, label, img_removed_150, pred_removed_150)
        )

        # =================================================
        # ADD 10% NOISE
        # =================================================
        img_noisy_10 = apply_gaussian_noise(orig_img, amount=0.10)
        pred_noisy_10 = predict_class(model, device, img_noisy_10)

        noisy_10_imgs.append(img_noisy_10)
        preds_noisy_10.append(pred_noisy_10)

        noise_10_results.append(
            (digit, orig_img, label, img_noisy_10, pred_noisy_10)
        )

        # =================================================
        # ADD 40% NOISE
        # =================================================
        img_noisy_40 = apply_gaussian_noise(orig_img, amount=0.40)
        pred_noisy_40 = predict_class(model, device, img_noisy_40)

        noisy_40_imgs.append(img_noisy_40)
        preds_noisy_40.append(pred_noisy_40)

        noise_40_results.append(
            (digit, orig_img, label, img_noisy_40, pred_noisy_40)
        )

    # -----------------------------------------------------
    # Flip rates
    # -----------------------------------------------------
    flip_5 = compute_flip_rate(remove_5_results)
    flip_150 = compute_flip_rate(remove_150_results)
    flip_10 = compute_flip_rate(noise_10_results)
    flip_40 = compute_flip_rate(noise_40_results)

    print("\nTC-CF-05 RESULTS:")
    print(f"Flip rate (5px removed):   {flip_5:.2f}%")
    print(f"Flip rate (150px removed): {flip_150:.2f}%")
    print(f"Flip rate (10% noise):     {flip_10:.2f}%")
    print(f"Flip rate (40% noise):     {flip_40:.2f}%")

    # -----------------------------------------------------
    # FIGURE 1 — Non-Salient Pixel Removal Results
    # -----------------------------------------------------
    fig1, ax1 = plt.subplots(10, 3, figsize=(8, 18))

    for i in range(10):
        # --- Column 1: Originals ---
        ax1[i, 0].imshow(originals[i], cmap="gray")
        ax1[i, 0].set_title(f"Original Digit {i}")
        ax1[i, 0].axis("off")

        # --- Column 2: Remove 5 ---
        ax1[i, 1].imshow(removed_5_imgs[i], cmap="gray")
        ax1[i, 1].set_title(f"Pred={preds_removed_5[i]}", fontsize=12)
        ax1[i, 1].axis("off")

        # --- Column 3: Remove 150 ---
        ax1[i, 2].imshow(removed_150_imgs[i], cmap="gray")
        ax1[i, 2].set_title(f"Pred={preds_removed_150[i]}", fontsize=12)
        ax1[i, 2].axis("off")

    fig1.suptitle(
        f"Pixel Removal Results\nFlip Rate (5px)={flip_5:.2f}% | Flip Rate (150px)={flip_150:.2f}%",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig("TC-CF-05_pixel_removal.png")
    plt.close(fig1)

    # -----------------------------------------------------
    # FIGURE 2 — Noise Results
    # -----------------------------------------------------
    fig2, ax2 = plt.subplots(10, 3, figsize=(8, 18))

    for i in range(10):
        # --- Column 1: Originals ---
        ax2[i, 0].imshow(originals[i], cmap="gray")
        ax2[i, 0].set_title(f"Original Digit {i}")
        ax2[i, 0].axis("off")

        # --- Column 2: Noise 10% ---
        ax2[i, 1].imshow(noisy_10_imgs[i], cmap="gray")
        ax2[i, 1].set_title(f"Pred={preds_noisy_10[i]}", fontsize=12)
        ax2[i, 1].axis("off")

        # --- Column 3: Noise 40% ---
        ax2[i, 2].imshow(noisy_40_imgs[i], cmap="gray")
        ax2[i, 2].set_title(f"Pred={preds_noisy_40[i]}", fontsize=12)
        ax2[i, 2].axis("off")

    fig2.suptitle(
        f"Noise Results\nFlip Rate (10% Noise)={flip_10:.2f}% | Flip Rate (40% Noise)={flip_40:.2f}%",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig("TC-CF-05_noise_results.png")
    plt.close(fig2)

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------
    assert flip_5 < 20, (
        f"High flip rate for removing 5 pixels: {flip_5:.2f}%"
    )

    assert flip_150 < 20, (
        f"High flip rate for removing 150 pixels: {flip_150:.2f}%"
    )

    assert flip_10 < 20, (
        f"High flip rate for 10% noise: {flip_10:.2f}%"
    )

    assert flip_40 < 20, (
        f"High flip rate for 40% noise: {flip_40:.2f}%"
    )


# ----------------------------
# TC-CF-06: OOD / Knowledge limits
# ----------------------------
def test_cf_ood_knowledge_limits():
    """
    TC-CF-06 OOD / Knowledge Limits Counterfactual Test:
    Input a non-digit character (letter 'A').
    Expected result:
        - Model should refuse to generate counterfactuals or produce only low-confidence ones.
        -  The model's softmax output shows low confidence, with no single class having a probability > 0.5.
    """
    # Setup device and load trained model
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)

    # Load trained model
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")

    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()

    # ------ Load OOD test image ('A') ------
    # Move up one directory, go into "test_images" folder, then pick the file
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    input_img_path = os.path.join(parent_dir, "test_images", "emnist_letter_A.png")
    # Uncomment below to show that the model is very confident in predicting 0
    # input_img_path = os.path.join(current_dir, "..", "test_images", "mnist_0.png")
    img = Image.open(input_img_path).convert("L")  # ensure grayscale

    # Resize to 28x28 if necessary
    if img.size != (28, 28):
        img = img.resize((28, 28))

    # Save test image
    test_img_path = os.path.join(current_dir, "TC-CF-06_ood_input_A.png")
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title("Out-of-Distribution Test Input ('A')")
    plt.axis("off")
    plt.savefig(test_img_path, bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()

    # Convert PIL image to PyTorch tensor and add batch dimension (1, 1, 28, 28)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        # Get raw model outputs (logits)
        output = model(img_tensor)
        # Convert logits → probabilities
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    # Print highest predicted probability
    max_prob = probs.max()
    pred_class = probs.argmax()
    print(f"Highest class: {pred_class} | Max softmax prob: {max_prob:.3f}")

    # --- Visualization: softmax probability distribution ---
    # This plot shows how confident the model is for each possible class (0–9).
    # Ideally, for an out-of-distribution input like 'A', the model should not
    # assign high confidence to any one class — all probabilities should be low
    # and relatively uniform.
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(probs)), probs, color="gray", alpha=0.7)

    # Highlight predicted (max) class
    bars[pred_class].set_color("red")

    # Add probability text labels to all bars
    for i, p in enumerate(probs):
        plt.text(i, p + 0.01, f"{p:.2f}", ha="center",
                 color="red" if i == pred_class else "dimgray",
                 fontweight="bold" if i == pred_class else "normal", fontsize=9)

    plt.title("Model Confidence Across Classes\n(Softmax Probability Distribution)", fontsize=13)
    plt.xlabel("Class Index", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.xticks(range(len(probs)), [str(i) for i in range(len(probs))])
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    # Save figure
    probs_fig_path = os.path.join(current_dir, "TC-CF_06_ood_softmax_distribution.png")
    plt.savefig(probs_fig_path, bbox_inches="tight", pad_inches=0.3, dpi=200)
    plt.close()

    print("Image tensor min/max:", img_tensor.min().item(), img_tensor.max().item())
    print("Predicted class:", pred_class)
    print("Max softmax probability:", max_prob)

    # Assert that the model is not overly confident on this out-of-distribution input.
    # If any class has > 0.5 probability, the model may be overconfident and fail this test.
    assert probs.max() < 0.5, f"Model too confident on OOD input (max={probs.max():.2f})."

    # Explanation of results:
    # The bars represent how confident the model is in each possible digit class (0–9).
    # When the model does not recognize the input (like an “A,” which isn’t a digit),
    # the probabilities should be roughly uniform — around 0.1 each for a 10-class classifier.
    # That indicates the model knows its limits — it’s not overconfident about an unfamiliar input.
    # If one bar spikes high (e.g., >0.5), that suggests the model is overconfident and possibly
    # not robust to out-of-distribution data.
