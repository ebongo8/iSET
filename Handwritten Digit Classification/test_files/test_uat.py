import torch
import torch.nn.functional as F
from captum.attr import Saliency
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from test_files.utils import get_model, get_device_type, load_trained_model
import os
import numpy as np


def test_meaningful_explanation():
    """
    TC-UAT-01 Meaningful Explanation Test:
    The user submits an image of a handwritten '3'. The system correctly classifies it
    and generates a saliency map.

    Expected result: The user confirms the saliency map highlights the curved strokes
    of the '3', making the decision process understandable.
    """

    # Setup
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)

    # Load trained model
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")

    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()

    # Load user-provided handwritten '3'
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    img_path = os.path.join(project_root, "test_images", "3.png")

    transparent_background = True
    inverted = True

    if transparent_background:
        img = Image.open(img_path).convert("RGBA")
        # Convert transparency to white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
        # Convert to grayscale
        img = img.convert("L")
    else:
        img = Image.open(img_path).convert("L")

    if inverted:
        # Invert colors (so digit becomes white, background black)
        img = ImageOps.invert(img)
        # Enhance contrast
        img = ImageOps.autocontrast(img)

    # Transform to match MNIST input
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Predict
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted digit: {pred_class}")

    # Generate saliency map
    saliency = Saliency(model)
    attr = saliency.attribute(img_tensor, target=pred_class)
    heatmap = attr.squeeze().abs().cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # normalize [0,1]

    # Prepare original image for overlay display
    img_for_display = img_tensor.squeeze().cpu().detach().numpy()
    img_for_display = img_for_display * 0.3081 + 0.1307  # reverse normalization

    # Save heatmap image
    plt.imshow(heatmap, cmap="hot")
    plt.axis("off")
    plt.title(f"Saliency Map for Predicted '{pred_class}'")
    plt.savefig("TC-UAT-01_heatmap_explanation_3.png")
    plt.close()

    # --- Side-by-side figure: input, overlay, heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # wide figure for titles

    # Processed input
    axes[0].imshow(img_for_display, cmap="gray")
    axes[0].set_title("Processed Input", fontsize=12)
    axes[0].axis("off")

    # Overlay of input + saliency
    axes[1].imshow(img_for_display, cmap="gray")
    axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
    axes[1].set_title(f"Saliency Map Overlay (Predicted: {pred_class})", fontsize=12)
    axes[1].axis("off")

    # Heatmap only
    axes[2].imshow(heatmap, cmap="hot")
    axes[2].set_title("Saliency Heatmap", fontsize=12)
    axes[2].axis("off")

    # Adjust layout to prevent title cutoff
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("TC-UAT-01_input_overlay_heatmap_3.png")
    plt.close()

    # Validate correct classification and explanation output
    assert pred_class == 3, f"Expected model to classify as '3', got '{pred_class}'"
    assert heatmap.shape == (28, 28), "Saliency heatmap not generated correctly."


def test_trust_building_failure_analysis():
    """
    TC-UAT-02 Trust Building via Failure Analysis:
    The system misclassifies a poorly written '9' as an '8'.
    The user examines the explanation.

    Expected result: The user can see that the model focused on the two closed loops, characteristic of an '8',
     providing a logical (though incorrect) reason for the failure. This helps the user understand the model's
     limitations and build appropriate trust.
    """
    # Setup
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)

    # Load trained model
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")

    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()

    # Load the poorly written "9" image
    img_path = os.path.join(project_root, "test_images", "poorly_written_9.png")
    img = Image.open(img_path).convert("L")
    img = ImageOps.autocontrast(img)  # enhance contrast

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Predict
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted class: {pred_class} (expected misclassification of '9' → '8')")

    # Generate saliency map
    saliency = Saliency(model)
    attr = saliency.attribute(img_tensor, target=pred_class)
    heatmap = attr.squeeze().abs().cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Visualization
    img_for_display = img_tensor.squeeze().cpu().detach().numpy()
    img_for_display = img_for_display * 0.3081 + 0.1307  # reverse normalization

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_for_display, cmap="gray")
    axes[0].set_title("Processed Input ('9')", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img_for_display, cmap="gray")
    axes[1].imshow(heatmap, cmap="hot", alpha=0.5)
    axes[1].set_title(f"Overlay (Predicted: {pred_class})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="hot")
    axes[2].set_title("Saliency Heatmap", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("TC-UAT-02_failure_analysis_9_to_8.png")
    plt.close()

    # Assertions
    # Expectation: model misclassifies '9' as '8'
    assert pred_class == 8, f"Expected misclassification as '8', got '{pred_class}'"

    # Heatmap check
    assert heatmap.shape == (28, 28), "Saliency heatmap not generated correctly."

    # Sanity check: ensure there’s strong saliency around the loop areas (upper + lower)
    upper_intensity = np.mean(heatmap[:14, :])
    lower_intensity = np.mean(heatmap[14:, :])
    avg_intensity = np.mean(heatmap)
    assert (upper_intensity > avg_intensity * 1.1) and (lower_intensity > avg_intensity * 1.1), \
        "Saliency does not highlight both loop regions—explanation not interpretable."
