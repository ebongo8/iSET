import torch
import pytest
import os
import numpy as np
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
def get_model_for_cf_tests():
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
    model, device = get_model_for_cf_tests()

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
        orig_image = load_digit_image(case["digit"])
        perturbed_image = apply_geometric_perturbation(orig_image, case["type"], case["params"])
        pred_label = predict_class(model, device, perturbed_image)
        proximity_delta = compute_proximity_delta(orig_image, perturbed_image)

        assert pred_label == case["label"], f"Flip detected for {case['desc']}"
        assert proximity_delta < 0.05, f"Proximity delta too high for {case['desc']}"


# ----------------------------
# TC-CF-02: Flip image
# ----------------------------
def test_tc_cf_02_flip_image():
    model, device = get_model_for_cf_tests()

    for d in range(10):
        # orig_image = load_digit_image(str(d))
        orig_image = get_mnist_image(index=11, train_data_set=True, show=True)
        flipped_image = flip_image(orig_image)
        pred_label = predict_class(model, device, flipped_image)
        assert pred_label == d, f"Prediction changed after horizontal flip for digit {d}"


# ----------------------------
# TC-CF-03: Brighten & strokes
# ----------------------------
def test_tc_cf_03_brightness_strokes():
    model, device = get_model_for_cf_tests()

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
    model, device = get_model_for_cf_tests()

    orig_image = load_digit_image("3")
    perturbed_image = apply_geometric_perturbation(orig_image, "straighten_arc", {})
    pred_label = predict_class(model, device, perturbed_image)
    assert pred_label == 3, "Ambiguous perturbation caused inconsistent decision"


# ----------------------------
# TC-CF-05: Blur
# ----------------------------
def test_tc_cf_05_blur():
    model, device = get_model_for_cf_tests()

    orig_image = load_digit_image("8")
    perturbed_image = blur_image(orig_image, sigma=1.0)
    pred_label = predict_class(model, device, perturbed_image)
    assert pred_label == 8, "Blur perturbation caused prediction flip"


# ----------------------------
# TC-CF-06: Noise
# ----------------------------
def test_tc_cf_06_noise():
    model, device = get_model_for_cf_tests()

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
    model, device = get_model_for_cf_tests()

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
