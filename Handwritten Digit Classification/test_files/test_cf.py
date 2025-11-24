import torch
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
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

    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 3 * n))

    fig.suptitle(
        f"TC-CF-01 Shape Sensitivity — Original vs Modified\n"
        f"Flip Rate = {flip_rate:.2f}%",
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
# TC-CF-03: Brighten & Strokes
# ----------------------------
def test_tc_cf_03_brightness_strokes():
    """
    TC-CF-03 Brighten & Strokes:

    1) Stroke thickening (digit 5):
       • Compute flip rate.

    2) Darken strokes by 10% (digits 0–9):
       • Compute flip rate.

    3) Brighten background (digits 0–9):
       • Compute flip rate.

    Metric:
       - Flip Rate: % of class flips after perturbation.
    """
    model, device = get_trained_model_for_cf_tests()

    # ============================================================
    # (1) STROKE THICKENING — FLIP RATE (digit "5")
    # ============================================================
    cf_test_images_dir = "../test_images/CF_images"
    orig_5_png = os.path.join(cf_test_images_dir, "mnist_5_24.png")
    thickened_5_png = os.path.join(cf_test_images_dir, "thicken_stroke_5.png")

    orig_5_arr = load_image(orig_5_png)
    thickened_5_arr = load_image(thickened_5_png)

    results = []
    pred_thick = predict_class(model, device, thickened_5_arr)

    results.append((0, orig_5_png, 5, thickened_5_arr, pred_thick))
    flip_rate_strokes = compute_flip_rate(results)

    assert flip_rate_strokes == 0, (
        f"Stroke thickening changed prediction! Flip Rate = {flip_rate_strokes:.1f}%"
    )

    # ============================================================
    # (2) DARKEN STROKES BY 10% — FLIP RATE (digits 0–9)
    # ============================================================
    flip_results_darken = []

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        true_label = digit

        dark_img = adjust_brightness(orig_img, factor=0.9)

        pred_orig = predict_class(model, device, orig_img)
        pred_dark = predict_class(model, device, dark_img)

        flip_results_darken.append((digit, None, true_label, dark_img, pred_dark))

    flip_rate_darken = compute_flip_rate(flip_results_darken)

    assert flip_rate_darken < 5, (
        f"[Darken 10%] Flip rate too high: {flip_rate_darken:.2f}%"
    )

    # ============================================================
    # (3) BRIGHTEN BACKGROUND — FLIP RATE (digits 0–9)
    # ============================================================
    flip_results_bright = []

    for digit in range(10):
        orig_img, _ = get_mnist_image(target_digit=digit, target_index=0, show=False)
        true_label = digit

        bright_img = adjust_brightness(orig_img, factor=1.1)

        pred_orig = predict_class(model, device, orig_img)
        pred_bright = predict_class(model, device, bright_img)

        flip_results_bright.append((digit, None, true_label, bright_img, pred_bright))

    flip_rate_bright = compute_flip_rate(flip_results_bright)

    assert flip_rate_bright < 5, (
        f"[Brighten BG] Flip rate too high: {flip_rate_bright:.2f}%"
    )

    # ============================================================
    # SUMMARY PRINT (does not affect pass/fail)
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

    # -------------------------
    # Load MNIST sample of "8"
    # -------------------------
    orig_img, label = get_mnist_image(target_digit=8, target_index=0, show=False)
    assert label == 8, "Loaded MNIST image is not the digit '8'"

    # -------------------------
    # Create a mask for the top loop
    # Approx region: upper half (rows 0–13)
    # You can tune this later
    # -------------------------
    mask = np.zeros_like(orig_img)
    mask[0:14, :] = 1.0

    # -------------------------
    # Apply Gaussian blur using your helper
    # -------------------------
    blurred_full = blur_image(orig_img, sigma=1.0)

    # Blend so only the masked region is blurred
    perturbed_img = orig_img * (1 - mask) + blurred_full * mask

    # -------------------------
    # Predictions
    # -------------------------
    pred_before = predict_class(model, device, orig_img)
    pred_after = predict_class(model, device, perturbed_img)

    # -------------------------
    # Visualization
    # -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(orig_img, cmap="gray")
    ax[0].set_title(f"Original (pred={pred_before})")
    ax[0].axis("off")

    ax[1].imshow(perturbed_img, cmap="gray")
    ax[1].set_title(f"Blurred Top Loop (pred={pred_after})")
    ax[1].axis("off")

    plt.tight_layout()
    fig.savefig("TC_CF_04_blur_before_after.png")
    plt.close(fig)

    # -------------------------
    # Assertion: prediction should remain stable
    # -------------------------
    assert pred_before == pred_after, (
        f"Localized blur caused prediction flip: "
        f"before={pred_before}, after={pred_after}"
    )


def get_random_non_salient_pixels(heatmap, num_pixels=5):
    """
    Return indices of 'num_pixels' random non-salient pixels
    based on the heatmap (low-value pixels = non-salient).
    """
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


# ----------------------------
# TC-CF-05: Noise
# ----------------------------
def test_tc_cf_05_noise():
    """
    TC-CF-05 Noise & Pixel Removal Robustness:
    For digits 0–9:
        • Remove 5 non-salient pixels (background).
        • Add 1–2% Gaussian noise.
    Evaluate:
        • Prediction stability
        • Flip rate
    """
    model, device = get_trained_model_for_cf_tests()

    remove_pixel_results = []   # (digit, orig_img, true_label, pert_img, pred_after)
    noise_results = []          # (digit, orig_img, true_label, pert_img, pred_after)

    originals = []
    removed_imgs = []
    noisy_imgs = []
    preds_removed = []
    preds_noisy = []

    percent_noise = 0.41

    # -----------------------------------------------------
    # Find LAST CONV layer for Grad-CAM
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
            target_digit=digit,
            target_index=0,
            show=False
        )
        assert label == digit

        # Store original
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

        # -----------------------------------------------------
        # Compute Grad-CAM heatmap (28×28)
        # -----------------------------------------------------
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
        # (A) REMOVE 5 NON-SALIENT BACKGROUND PIXELS
        # =================================================
        # Get 5 random non-salient pixels
        idxs = get_random_non_salient_pixels(heatmap_t, num_pixels=500)
        img_removed = orig_img.copy().flatten()
        img_removed[idxs.numpy()] = 0.0
        img_removed = img_removed.reshape(orig_img.shape)
        pred_removed = predict_class(model, device, img_removed)

        removed_imgs.append(img_removed)
        preds_removed.append(pred_removed)

        remove_pixel_results.append((digit, orig_img, label, img_removed, pred_removed))

        # =================================================
        # (B) ADD 10% GAUSSIAN NOISE
        # =================================================
        img_noisy = add_noise(orig_img, amount=percent_noise)
        pred_noisy = predict_class(model, device, img_noisy)

        noisy_imgs.append(img_noisy)
        preds_noisy.append(pred_noisy)

        noise_results.append(
            (digit, orig_img, label, img_noisy, pred_noisy)
        )

    # -----------------------------------------------------
    # Flip rates
    # -----------------------------------------------------
    remove_flip_rate = compute_flip_rate(remove_pixel_results)
    noise_flip_rate = compute_flip_rate(noise_results)

    print("\nTC-CF-05 RESULTS:")
    print(f"Flip rate (remove 5 pixels): {remove_flip_rate:.2f}%")
    print(f"Flip rate ({percent_noise*100:.2f}% noise):      {noise_flip_rate:.2f}%")

    # -----------------------------------------------------
    # Visualization Grid (10 rows × 3 columns)
    # -----------------------------------------------------
    fig, axes = plt.subplots(10, 3, figsize=(9, 18))

    for i in range(10):
        # --- Column 1: Original ---
        axes[i, 0].imshow(originals[i], cmap="gray")
        axes[i, 0].set_title(f"Digit {i}\nOriginal")
        axes[i, 0].axis("off")

        # --- Column 2: Pixel Removal ---
        axes[i, 1].imshow(removed_imgs[i], cmap="gray")
        axes[i, 1].set_title(f"Remove 5 px\nPred={preds_removed[i]}")
        axes[i, 1].axis("off")

        # --- Column 3: Noise ---
        axes[i, 2].imshow(noisy_imgs[i], cmap="gray")
        axes[i, 2].set_title(f"Noise 2%\nPred={preds_noisy[i]}")
        axes[i, 2].axis("off")

    # Title showing flip rates
    fig.suptitle(
        f"TC-CF-05 Noise & Pixel Removal\n"
        f"Flip Rate (remove 5 px): {remove_flip_rate:.2f}%   |   "
        f"Flip Rate (noise {percent_noise*100:.2f}%): {noise_flip_rate:.2f}%",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("TC_CF_05_noise_results.png")
    plt.close(fig)

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------
    assert remove_flip_rate < 20, (
        f"High flip rate for removing 5 pixels: {remove_flip_rate:.2f}%"
    )
    assert noise_flip_rate < 20, (
        f"High flip rate for {percent_noise}% noise: {noise_flip_rate:.2f}%"
    )


# ----------------------------
# TC-CF-06: OOD / Knowledge limits
# ----------------------------
def test_tc_cf_06_ood():
    model, device = get_trained_model_for_cf_tests()

    # Create simple 'A' placeholder as OOD input
    ood_image = np.zeros((28, 28))
    ood_image[5:23, 10:18] = 1.0

    # Forward pass
    img_tensor = torch.tensor(ood_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    output = model(img_tensor)

    # Predicted class
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted class for OOD input: {pred_class}")

    # Confidence (max softmax)
    confidence = torch.softmax(output, dim=1).max().item()
    print(f"Model confidence on OOD input: {confidence:.3f}")

    # Generate CF
    cf = generate_counterfactual(ood_image, model)

    # Assertions
    assert confidence < 0.5, "Model overconfident on OOD input"
    assert cf is None, "CF generator produced unsafe CF for OOD input"
