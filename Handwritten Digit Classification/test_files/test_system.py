import torch
import numpy as np
from captum.attr import Saliency, LayerGradCam
from test_files.utils import get_model, get_dataloaders, get_device_type, load_trained_model
import pytest
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def test_performance():
    """
    TC-ST-01  Performance Test: Train the model for 10 epochs. Evaluate on the entire unseen MNIST test set.
    Expected result: Accuracy is >= 95%. A confusion matrix is generated showing class-wise performance.
    """
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # Load trained model (already saved in src/model_state.pt)
    # Load the trained model
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")
    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()
    _, test_loader = get_dataloaders()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")

    assert accuracy >= 0.95, f"Expected >=95% accuracy, got {accuracy:.2%}"


def test_explanation_generation():
    """
    TC-ST-02 Explanation Generation Test: For a correctly classified image from the test set,
    generate a saliency map using Grad-CAM.
    Expected result: A 28x28 heatmap image is successfully generated without errors.
    """
    # Setup device and load trained model
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)

    # Load trained model (already saved in src/model_state.pt)
    # Load the trained model
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    path_to_saved_model = os.path.join(project_root, "src", "model_state.pt")

    model = load_trained_model(path_to_saved_model, device_type)
    model.to(device)
    model.eval()

    # Load test data
    _, test_loader = get_dataloaders()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    # Select a correctly classified sample of a specific number
    desired_label = 7
    correct_indices = (preds == labels).nonzero(as_tuple=True)[0]
    specific_indices = [i.item() for i in correct_indices if labels[i] == desired_label]
    assert len(specific_indices) > 0, f"No correctly classified images of {desired_label} found."
    idx = specific_indices[0]  # pick the first matching sample
    image = images[idx].unsqueeze(0)
    target_class = preds[idx].item()

    # Find last convolutional layer dynamically
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = layer
    assert last_conv_layer is not None, "No Conv2d layer found for Grad-CAM."

    # Grad-CAM
    gradcam = LayerGradCam(model, last_conv_layer)
    attr = gradcam.attribute(image, target=target_class)

    # Ensure correct shape before interpolation
    if attr.dim() == 4:
        attr = torch.nn.functional.interpolate(attr, size=(28, 28), mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Expected 4D Grad-CAM output, got shape {attr.shape}")

    # Convert to numpy heatmap
    heatmap = attr.squeeze().cpu().detach().numpy()
    assert heatmap.shape == (28, 28), "Expected 28x28 Grad-CAM heatmap."

    # Save visualization
    plt.imshow(heatmap, cmap="hot")
    plt.axis("off")
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.title("Grad-CAM Heatmap")
    plt.savefig("gradcam_heatmap.png", bbox_inches='tight')

    assert attr.shape == (1, 1, 28, 28), "Expected attribution shape (1, 1, 28, 28)."


def test_explanation_accuracy():
    """
    TC-ST-03 Explanation Accuracy Test: Identify the top 20% most salient pixels from TC-ST-02.
    Mask these pixels (set to 0) and re-run prediction. Repeat for 20% random pixels.
    Expected result: The prediction confidence (softmax output) drops significantly more when
    masking salient pixels vs. random pixels.
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

    # Load test data
    _, test_loader = get_dataloaders()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    # Select a correctly classified sample (same as TC-ST-02)
    desired_label = 7
    correct_indices = (preds == labels).nonzero(as_tuple=True)[0]
    specific_indices = [i.item() for i in correct_indices if labels[i] == desired_label]
    assert len(specific_indices) > 0, f"No correctly classified images of {desired_label} found."

    idx = specific_indices[0]  # pick the first matching sample
    image = images[idx].unsqueeze(0)
    target_class = preds[idx].item()

    plt.imshow(image.squeeze().cpu(), cmap="gray")
    plt.title(f"Original Image (Label={desired_label}, Pred={target_class})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"original_image_label{desired_label}.png", bbox_inches="tight")
    plt.show()

    # Get original confidence
    with torch.no_grad():
        orig_output = model(image)
        orig_conf = torch.nn.functional.softmax(orig_output, dim=1)[0, target_class].item()

    # Load Grad-CAM heatmap from TC-ST-02
    heatmap = plt.imread("gradcam_heatmap.png")
    if heatmap.ndim == 3:  # handle RGB PNGs
        heatmap = heatmap[..., 0]
    heatmap = torch.tensor(heatmap, dtype=torch.float32)

    # Normalize heatmap between 0 and 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Flatten and get top 20% salient pixel indices
    flat_heatmap = heatmap.flatten()
    k = int(0.2 * flat_heatmap.numel())
    topk_indices = torch.topk(flat_heatmap, k).indices

    # Mask 20% salient pixels (set to 0)
    image_masked = image.clone()
    flat_img = image_masked.flatten().clone()
    flat_img[topk_indices] = 0.0
    image_masked = flat_img.view_as(image_masked)

    # Re-run prediction on masked image
    with torch.no_grad():
        out_salient_masked = model(image_masked)
        # The model’s confidence in the original class after erasing the important pixels
        conf_salient_masked = torch.nn.functional.softmax(out_salient_masked, dim=1)[0, target_class].item()

    # Mask 20% random pixels for comparison
    rand_indices = torch.randperm(flat_heatmap.numel())[:k]
    image_random_masked = image.clone()
    flat_img_rand = image_random_masked.flatten().clone()
    flat_img_rand[rand_indices] = 0.0
    image_random_masked = flat_img_rand.view_as(image_random_masked)

    with torch.no_grad():
        out_random_masked = model(image_random_masked)
        conf_random_masked = torch.nn.functional.softmax(out_random_masked, dim=1)[0, target_class].item()

    # Assertions / Expectations
    drop_salient = orig_conf - conf_salient_masked
    drop_random = orig_conf - conf_random_masked

    # Visual the images and confidences
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axs[0].imshow(image.squeeze().cpu(), cmap="gray")
    axs[0].axis("off")
    axs[0].text(
        0.5,-0.05, f"Conf={orig_conf:.3f}", ha='center', va='top', transform=axs[0].transAxes, fontsize=12
    )
    axs[0].set_title(f"Original (Label={desired_label})")

    # Masked salient pixels
    axs[1].imshow(image_masked.squeeze().cpu(), cmap="gray")
    axs[1].axis("off")
    axs[1].text(
        0.5, -0.05, f"Conf={conf_salient_masked:.3f}", ha='center', va='top', transform=axs[1].transAxes, fontsize=12
    )
    axs[1].set_title("Salient Mask")

    # Masked random pixels
    axs[2].imshow(image_random_masked.squeeze().cpu(), cmap="gray")
    axs[2].axis("off")
    axs[2].text(
        0.5, -0.05, f"Conf={conf_random_masked:.3f}", ha='center', va='top', transform=axs[2].transAxes, fontsize=12
    )
    axs[2].set_title("Random Mask")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig("masked_image_comparison.png", bbox_inches='tight')
    plt.show()

    print(f"Original confidence: {orig_conf:.5f}")
    print(f"Confidence after salient mask: {conf_salient_masked:.5f}")
    print(f"Confidence after random mask: {conf_random_masked:.5f}")
    print(f"Drop (salient): {drop_salient:.5f}, Drop (random): {drop_random:.5f}")

    # Expected: confidence drop is greater for salient pixels
    # The model’s confidence should fall more when you remove important pixels than when you remove random ones.
    assert drop_salient > drop_random, (
        f"Expected confidence drop to be larger for salient pixels "
        f"({drop_salient:.4f}) than for random pixels ({drop_random:.4f})."
    )


def test_counterfactual_robustness():
    """
    TC-ST-04 Counterfactual Robustness Test: Apply a small adversarial perturbation to a correctly classified image.
    Expected result: The model's prediction should remain unchanged. A flipped prediction indicates a vulnerability.
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
    _, test_loader = get_dataloaders()
    model.eval()

    # Load test data
    images, labels = next(iter(test_loader))
    image = images[0].unsqueeze(0).to(device)
    label = labels[0].unsqueeze(0).to(device)

    image.requires_grad = True
    output = model(image)
    pred_class = output.argmax(dim=1)
    loss = F.cross_entropy(output, pred_class)
    loss.backward()

    epsilon = 0.1
    perturbation = epsilon * image.grad.sign()
    adv_image = torch.clamp(image + perturbation, 0, 1)

    adv_output = model(adv_image)
    adv_pred = adv_output.argmax(dim=1)

    # --- Visualization ---
    # Convert tensors to numpy
    orig_img = image.detach().cpu().squeeze().numpy()
    pert_img = adv_image.detach().cpu().squeeze().numpy()

    # --- Create side-by-side figure with a border ---
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(orig_img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(pert_img, cmap="gray")
    axes[1].set_title("Perturbed Image")
    axes[1].axis("off")

    # Add a little breathing room between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.tight_layout(pad=2.0)

    # --- Save figure with padding/border ---
    save_path = os.path.join(current_dir, "original_vs_perturbed.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()

    print(f"Original: {pred_class.item()}, Adversarial: {adv_pred.item()}")

    assert adv_pred == pred_class, "Prediction changed after small perturbation — model not robust."


def test_knowledge_limits():
    """
    TC-ST-05 Knowledge Limits Test: Input an image of the letter 'A'.
    Expected result: The model's softmax output shows low confidence,
    with no single class having a probability > 0.5.
    """
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    model = get_model()
    model.eval()

    # Create a blank 28x28 grayscale (mode 'L') image with black background (color=0)
    img = Image.new("L", (28, 28), color=0)
    # Draw a white letter "A" near the top-left of the image
    draw = ImageDraw.Draw(img)
    draw.text((4, 0), "A", fill=255)
    # Visual confirmation of the test image
    plt.imshow(img, cmap="gray")
    plt.title("OOD Test Input ('A')")
    plt.axis("off")
    plt.show()

    # Convert PIL image to PyTorch tensor and add batch dimension (1, 1, 28, 28)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get raw model outputs (logits)
        output = model(img_tensor)
        # Convert logits → probabilities
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    # Print highest predicted probability
    print(f"Max softmax prob: {probs.max():.3f}")
    # Assert that the model is not overly confident on this out-of-distribution input.
    # If any class has > 0.5 probability, the model may be overconfident and fail this test.
    assert probs.max() < 0.5, f"Model too confident on OOD input (max={probs.max():.2f})."
