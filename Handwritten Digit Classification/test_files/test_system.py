import torch
import numpy as np
# from captum.attr import Saliency, LayerGradCam
from test_files.utils import get_model, get_dataloaders, get_device_type
import pytest
# import torch.nn.functional as F
# from torchvision import transforms
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt


def test_performance():
    """
    TC-ST-01  Performance Test: Train the model for 10 epochs. Evaluate on the entire unseen MNIST test set.
    Expected result: Accuracy is >= 95%. A confusion matrix is generated showing class-wise performance.
    """
    # TODO Heather review/update code below
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)
    model = get_model()
    _, test_loader = get_dataloaders()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    assert accuracy >= 0.95, f"Expected >=95% accuracy, got {accuracy:.2%}"


def test_explanation_generation():
    """
    TC-ST-02 Explanation Generation Test: For a correctly classified image from the test set,
    generate a saliency map using Grad-CAM.
    Expected result: A 28x28 heatmap image is successfully generated without errors.
    """
    # TODO Erin review/update code below
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)
    model = get_model()
    _, test_loader = get_dataloaders()
    saliency = Saliency(model)

    images, labels = next(iter(test_loader))
    image = images[0].unsqueeze(0).to(device)
    # images, labels = next(iter(test_loader))
    # images, labels = images.to(device), labels.to(device)
    image.requires_grad = True
    output = model(image)
    pred_class = output.argmax(dim=1)

    attr = saliency.attribute(image, target=pred_class)

    # idx = (pred == labels).nonzero(as_tuple=True)[0][0]
    # image = images[idx].unsqueeze(0)
    # target = pred[idx].item()
    #
    # gradcam = LayerGradCam(model, model.features[-1] if hasattr(model, "features") else list(model.children())[-1])
    # attr = gradcam.attribute(image, target=target)
    # attr = torch.nn.functional.interpolate(attr, size=(28, 28), mode="bilinear")
    # heatmap = attr.squeeze().cpu().detach().numpy()
    #
    # plt.imshow(heatmap, cmap="hot")
    # plt.axis("off")
    # plt.title("Grad-CAM Heatmap")
    # plt.savefig("gradcam_heatmap.png")
    #
    # assert heatmap.shape == (28, 28), "Expected 28x28 Grad-CAM heatmap."

    assert attr.shape == (1, 1, 28, 28)


def test_explanation_accuracy():
    """
    TC-ST-03 Explanation Accuracy Test: Identify the top 20% most salient pixels from TC-ST-02.
    Mask these pixels (set to 0) and re-run prediction. Repeat for 20% random pixels.
    Expected result: The prediction confidence (softmax output) drops significantly more when
    masking salient pixels vs. random pixels.
    """
    # TODO Erin review/update code below
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)

    model = get_model()
    _, test_loader = get_dataloaders()
    model.eval()

    saliency = Saliency(model)
    images, labels = next(iter(test_loader))
    image = images[0].unsqueeze(0).to(device)
    label = labels[0].item()

    output = model(image)
    pred_class = output.argmax(dim=1).item()
    if pred_class != label:
        pytest.skip("Skipping: first image not correctly classified.")

    attr = saliency.attribute(image, target=pred_class).abs().squeeze().cpu().numpy()
    flat = attr.flatten()
    threshold = np.percentile(flat, 80)

    salient_mask = (attr >= threshold)
    random_mask = np.random.rand(*attr.shape) >= 0.8

    image_np = image.squeeze().cpu().numpy()
    salient_masked = image_np.copy()
    random_masked = image_np.copy()
    salient_masked[salient_mask] = 0
    random_masked[random_mask] = 0

    def get_conf(img):
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
        out = model(tensor)
        return F.softmax(out, dim=1)[0, pred_class].item()

    orig_conf = get_conf(image_np)
    salient_conf = get_conf(salient_masked)
    random_conf = get_conf(random_masked)

    print(f"Original: {orig_conf:.3f}, Salient masked: {salient_conf:.3f}, Random masked: {random_conf:.3f}")

    assert salient_conf < random_conf, \
        f"Expected salient masking to reduce confidence more (salient={salient_conf:.3f}, random={random_conf:.3f})."


def test_counterfactual_robustness():
    """
    TC-ST-04 Counterfactual Robustness Test: Apply a small adversarial perturbation to a correctly classified image.
    Expected result: The model's prediction should remain unchanged. A flipped prediction indicates a vulnerability.
    """
    # TODO Heather review/update code below
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)
    model = get_model()
    _, test_loader = get_dataloaders()
    model.eval()

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

    print(f"Original: {pred_class.item()}, Adversarial: {adv_pred.item()}")

    assert adv_pred == pred_class, "Prediction changed after small perturbation — model not robust."


def test_knowledge_limits():
    """
    TC-ST-05 Knowledge Limits Test: Input an image of the letter 'A'.
    Expected result: The model's softmax output shows low confidence, with no single class having a probability > 0.5.
    """
    # TODO Erin review/update code below
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)

    model = get_model()
    model.eval()

    from PIL import Image, ImageDraw
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((4, 0), "A", fill=255)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    print(f"Max softmax prob: {probs.max():.3f}")
    assert probs.max() < 0.5, f"Model too confident on OOD input (max={probs.max():.2f})."
