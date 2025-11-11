import torch
import torch.nn.functional as F
from captum.attr import Saliency, LayerGradCam
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from test_files.utils import get_device_type, load_trained_model
import os
import numpy as np


# ============================================================
# Helper Functions
# ============================================================
def load_and_preprocess_image(img_path, device, transparent_background=False, invert=False):
    """Load image and convert to MNIST tensor"""
    if transparent_background:
        img = Image.open(img_path).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("L")
    else:
        img = Image.open(img_path).convert("L")

    if invert:
        img = ImageOps.invert(img)
        img = ImageOps.autocontrast(img)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True
    return img_tensor, img


def generate_heatmap(attr):
    """Normalize attribution map to [0,1]"""
    heatmap = attr.squeeze().abs().cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


def find_last_conv_layer(model):
    """Find last convolutional layer in model for Grad-CAM"""
    last_conv = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model for Grad-CAM.")
    return last_conv


def create_saliency_and_gradcam_heatmaps(model, img_tensor, target_class):
    """Compute Saliency and Grad-CAM heatmaps"""
    saliency = Saliency(model)
    attr_saliency = saliency.attribute(img_tensor, target=target_class)
    heatmap_saliency = generate_heatmap(attr_saliency)

    last_conv = find_last_conv_layer(model)
    gradcam = LayerGradCam(model, last_conv)
    attr_gc = gradcam.attribute(img_tensor, target=target_class)
    attr_gc = F.interpolate(attr_gc, size=(28,28), mode="bilinear", align_corners=False)
    heatmap_gradcam = generate_heatmap(attr_gc)

    return heatmap_saliency, heatmap_gradcam


def create_output_visualization(img_tensor, heatmap_saliency, heatmap_gradcam, pred_class, output_path, title_suffix=""):
    """3-row visualization: original, saliency, grad-cam"""
    img_for_display = img_tensor.squeeze().cpu().detach().numpy()
    img_for_display = img_for_display * 0.3081 + 0.1307

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])  # 3 rows, 2 columns

    # Top row: original input spans both columns
    ax_orig = fig.add_subplot(gs[0, :])
    ax_orig.imshow(img_for_display, cmap="gray")
    ax_orig.set_title(f"Processed Input {title_suffix}", fontsize=14, pad=10)
    ax_orig.axis("off")

    # Row 2: saliency overlay + raw
    ax_sal_overlay = fig.add_subplot(gs[1, 0])
    ax_sal_overlay.imshow(img_for_display, cmap="gray")
    ax_sal_overlay.imshow(heatmap_saliency, cmap="hot", alpha=0.5)
    ax_sal_overlay.set_title(f"Saliency Overlay (Predicted: {pred_class})", fontsize=12)
    ax_sal_overlay.axis("off")

    ax_sal_map = fig.add_subplot(gs[1, 1])
    ax_sal_map.imshow(heatmap_saliency, cmap="hot")
    ax_sal_map.set_title("Saliency Heatmap", fontsize=12)
    ax_sal_map.axis("off")

    # Row 3: grad-cam overlay + raw
    ax_grad_overlay = fig.add_subplot(gs[2, 0])
    ax_grad_overlay.imshow(img_for_display, cmap="gray")
    ax_grad_overlay.imshow(heatmap_gradcam, cmap="hot", alpha=0.5)
    ax_grad_overlay.set_title(f"Grad-CAM Overlay (Predicted: {pred_class})", fontsize=12)
    ax_grad_overlay.axis("off")

    ax_grad_map = fig.add_subplot(gs[2, 1])
    ax_grad_map.imshow(heatmap_gradcam, cmap="hot")
    ax_grad_map.set_title("Grad-CAM Heatmap", fontsize=12)
    ax_grad_map.axis("off")

    plt.subplots_adjust(wspace=0.15, hspace=0.25, left=0.05, right=0.95, top=0.93, bottom=0.05)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


# ============================================================
# Tests
# ============================================================

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
    img_tensor, img = load_and_preprocess_image(
        os.path.join(project_root, "test_images", "3.png"),
        device,
        transparent_background=True,
        invert=True
    )

    # Predict
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted digit: {pred_class}")

    # Compute heatmaps
    heatmap_saliency, heatmap_gradcam = create_saliency_and_gradcam_heatmaps(model, img_tensor, pred_class)

    # Visualize
    create_output_visualization(
        img_tensor, heatmap_saliency, heatmap_gradcam, pred_class,
        "TC-UAT-01_input_overlay_heatmap_3.png", title_suffix="'3'"
    )

    # Validate correct classification and explanation output
    assert pred_class == 3, f"Expected model to classify as '3', got '{pred_class}'"
    assert heatmap_saliency.shape == (28, 28), "Saliency heatmap not generated correctly."
    assert heatmap_gradcam.shape == (28, 28), "GradCam heatmap not generated correctly."


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

    img_tensor, img = load_and_preprocess_image(
        os.path.join(project_root, "test_images", "poorly_written_9.png"),
        device
    )

    # Predict
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted class: {pred_class} (expected misclassification of '9' → '8')")

    # Compute heatmaps
    heatmap_saliency, heatmap_gradcam = create_saliency_and_gradcam_heatmaps(model, img_tensor, pred_class)

    # Visualize
    create_output_visualization(
        img_tensor, heatmap_saliency, heatmap_gradcam, pred_class,
        "TC-UAT-02_failure_analysis_9_to_8.png", title_suffix="'9'"
    )

    # Assertions
    assert pred_class == 8, f"Expected misclassification as '8', got '{pred_class}'"
    assert heatmap_saliency.shape == (28, 28), "Saliency heatmap not generated correctly."
    assert heatmap_gradcam.shape == (28, 28), "GradCam heatmap not generated correctly."

    upper_intensity = np.mean(heatmap_saliency[:14, :])
    lower_intensity = np.mean(heatmap_saliency[14:, :])
    avg_intensity = np.mean(heatmap_saliency)
    assert (upper_intensity > avg_intensity * 1.1) and (lower_intensity > avg_intensity * 1.1), \
        "Saliency does not highlight both loop regions—explanation not interpretable."
