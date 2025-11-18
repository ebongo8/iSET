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
    generate_counterfactual,
    compute_proximity_delta,
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
    #   CREATE SUMMARY PNG
    # ---------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 3 * n))
    fig.suptitle("TC-CF-01 Shape Sensitivity — Original vs Modified", fontsize=18)

    if n == 1:
        axes = np.array([axes])  # Ensure consistent 2D indexing

    for i, res in enumerate(results):
        ax_orig = axes[i, 0]
        ax_mod = axes[i, 1]

        # Original
        ax_orig.imshow(res["orig_img"], cmap="gray")
        ax_orig.set_title(
            f"Original: {res['orig_fname']}\nDigit {res['true']}"
        )
        ax_orig.axis("off")

        # Modified + results
        status_pred = "PASS" if res["pass_pred"] else "FAIL"
        status_prox = "PASS" if res["pass_prox"] else "FAIL"

        ax_mod.imshow(res["mod_img"], cmap="gray")
        ax_mod.set_title(
            f"Predicted: {res['pred']} ({status_pred})\n"
            f"L1 Δ = {res['prox']['L1']:.4f}, L2 Δ = {res['prox']['L2']:.4f} ({status_prox})"
        )
        ax_mod.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("TC-CF-01_shape_sensitivity_summary.png")
    plt.close()

    failures = [
        res for res in results
        if not res["pass_pred"] or not res["pass_prox"]
    ]
    # Remove img arrays from res dictionaries so they don't print if assertion fails
    for res in failures:
        res.pop("orig_img")
        res.pop("mod_img")

    assert not failures, (
        "TC-CF-01 failed for:\n" +
        "\n".join(
            f"{res['prefix']}  true={res['true']} pred={res['pred']}  "
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

    # ------------------------------------------------------------
    # Compute flip rate
    # ------------------------------------------------------------
    total = len(results)
    num_flipped = sum(1 for _, _, y, _, yhat in results if yhat != y)
    flip_rate = (num_flipped / total) * 100

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
# TC-CF-03: Brighten & strokes
# ----------------------------
def test_tc_cf_03_brightness_strokes():
    model, device = get_trained_model_for_cf_tests()

    test_cases = [
        {"desc": "Thicken middle stroke of 5", "digit": "5", "label": 5, "type": "dilate_stroke",
         "params": {"factor": 1}},
        {"desc": "Darken strokes by 10%", "digit": "5", "label": 5, "type": "brightness",
         "params": {"factor": 0.9}},
        {"desc": "Brighten background", "digit": "5", "label": 5, "type": "brightness",
         "params": {"factor": 1.1}},
    ]

    for case in test_cases:
        orig_image = load_digit_image(case["digit"])
        if case["type"] == "dilate_stroke":
            perturbed_image = apply_geometric_perturbation(orig_image, "dilate_stroke", case["params"])
        else:
            perturbed_image = adjust_brightness(orig_image, case["params"]["factor"])
        pred_label = predict_class(model, device, perturbed_image)
        assert pred_label == case["label"], f"Prediction flipped for {case['desc']}"


# ----------------------------
# TC-CF-04: Ambiguous decision
# ----------------------------
def test_tc_cf_04_ambiguous_decision():
    model, device = get_trained_model_for_cf_tests()

    orig_image = load_digit_image("3")
    perturbed_image = apply_geometric_perturbation(orig_image, "straighten_arc", {})
    pred_label = predict_class(model, device, perturbed_image)
    assert pred_label == 3, "Ambiguous perturbation caused inconsistent decision"


# ----------------------------
# TC-CF-05: Blur
# ----------------------------
def test_tc_cf_05_blur():
    model, device = get_trained_model_for_cf_tests()

    orig_image = load_digit_image("8")
    perturbed_image = blur_image(orig_image, sigma=1.0)
    pred_label = predict_class(model, device, perturbed_image)
    assert pred_label == 8, "Blur perturbation caused prediction flip"


# ----------------------------
# TC-CF-06: Noise
# ----------------------------
def test_tc_cf_06_noise():
    model, device = get_trained_model_for_cf_tests()

    # Remove 5 non-salient pixels
    orig_image = load_digit_image("5")
    perturbed_image = apply_geometric_perturbation(orig_image, "remove_random_pixels", {"count": 5})
    pred_label = predict_class(model, device, perturbed_image)
    assert pred_label == 5, "Removing non-salient pixels caused flip"

    # Add random noise 1-2%
    orig_image = load_digit_image("5")
    perturbed_image = add_noise(orig_image, amount=0.02)
    pred_label, _ = model.predict(perturbed_image)
    assert pred_label == 5, "Adding background noise caused flip"


# ----------------------------
# TC-CF-07: OOD / Knowledge limits
# ----------------------------
def test_tc_cf_07_ood():
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
