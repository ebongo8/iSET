import torch
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from test_files.utils import (
    get_device_type,
    load_trained_model,
    get_mnist_image,
    apply_geometric_perturbation,
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


def predict_class(model, device, img):
    """
    Converts a numpy image to tensor, forwards through model, returns predicted class.
    """
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    return pred_class


# ----------------------------
# TC-CF-01: Shape sensitivity
# ----------------------------
def test_tc_cf_01_shape_sensitivity():
    model, device = get_trained_model_for_cf_tests()

    test_cases = [
        {"desc": "Close top loop of 3", "digit": "3", "label": 3, "type": "close_loop",
         "params": {"width": 1}},
        {"desc": "Break bottom loop of 8", "digit": "8", "label": 8, "type": "break_loop",
         "params": {"width": 1}},
        {"desc": "Erase vertical line on 4", "digit": "4", "label": 4, "type": "erase_line",
         "params": {}},
        {"desc": "Add line in middle of 1", "digit": "1", "label": 1, "type": "add_line",
         "params": {"width": 1}},
        {"desc": "Connect top and bottom loops of 0", "digit": "0", "label": 0, "type": "connect_loops",
         "params": {"width": 1}},
        {"desc": "Extend tail on 9", "digit": "9", "label": 9, "type": "extend_tail",
         "params": {"length": 2}},
        {"desc": "Straighten top arc of 3", "digit": "3", "label": 3, "type": "straighten_arc",
         "params": {}},
        {"desc": "Open bottom loop of 6", "digit": "6", "label": 6, "type": "open_loop",
         "params": {"width": 1}},
    ]

    for case in test_cases:
        orig_image, _ = get_mnist_image(target_digit=case["digit"],index=None, train_data_set=True, show=True)
        perturbed_image = apply_geometric_perturbation(orig_image, case["type"], case["params"])
        pred_label = predict_class(model, device, perturbed_image)
        proximity_delta = compute_proximity_delta(orig_image, perturbed_image)

        assert pred_label == case["label"], f"Flip detected for {case['desc']}"
        assert proximity_delta < 0.05, f"Proximity delta too high for {case['desc']}"


# ----------------------------
# TC-CF-02: Flip image
# ----------------------------
def test_tc_cf_02_flip_image():
    """
    Flip Image: Measure spatial invariance by horizontally mirroring input digits. Counterfactual metric outputted is stability for analysis
    :return:
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

    # ---------------------------------------------------------------------
    # Compute summary statistics
    # ---------------------------------------------------------------------
    total = len(results)
    num_correct = sum(1 for _, _, y, _, yhat in results if y == yhat)
    stability = (num_correct / total) * 100

    # ---------------------------------------------------------------------
    # Create the PNG summary for all processed indices
    # ---------------------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(6, 2 * n))
    fig.suptitle(
        f"TC-CF-02 Flip Image Results\nStability: {stability:.1f}%",
        fontsize=16
    )
    for row, (idx, orig_img, true_label, flip_img, pred_label) in enumerate(results):
        ax_orig = axes[row, 0] if n > 1 else axes[0]
        ax_flip = axes[row, 1] if n > 1 else axes[1]

        ax_orig.imshow(orig_img, cmap="gray")
        ax_orig.set_title(f"Original (Label: {true_label})")
        ax_orig.axis("off")

        ax_flip.imshow(flip_img, cmap="gray")
        ax_flip.set_title(f"Flipped (Pred: {pred_label})")
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
