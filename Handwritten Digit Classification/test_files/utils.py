import torch
from src.classifier_model import ImageClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, binary_dilation
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import matplotlib.pyplot as plt


def get_device_type(windows_os=False):
    if windows_os:
        # return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return "mps" if torch.backends.mps.is_available() else "cpu"


def get_model():
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    return ImageClassifier().to(device)


def load_trained_model(path_to_saved_model, device_type):
    device = torch.device(device_type)
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(path_to_saved_model, map_location=device_type))
    model.eval()
    return model


def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # return (
    #     torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
    #     torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False),
    # )
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        DataLoader(test_dataset, batch_size=32, shuffle=False),
    )


def prepare_image(image_path, device, img_size=(28, 28)):
    """
    Preprocess an image so it's compatible with a model expecting grayscale input.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to send the tensor to.
        img_size (tuple): Target size (height, width) expected by the model.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for classification.
    """

    # Load the image
    img = Image.open(image_path).convert('L')  # convert to grayscale (1 channel)

    # Define transform: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # same normalization used during training
    ])

    # Apply transforms
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

    return img_tensor


def find_index_for_label(label_input):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    # find indices of all samples labeled '9'
    indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == label_input]

    print("Indices with label 9:", indices[:10])  # show first 10 for reference


def get_mnist_image(target_digit=None, index=None, train_data_set=True, show=True):
    """
    Load a single MNIST sample either by dataset index OR by specifying the digit (0-9).

    Args:
        target_digit (int or None): The digit you want to load (0–9). If provided,
                                    the function finds the first MNIST example of this digit.
        index (int or None): If provided, loads the MNIST sample by index instead.
                             If both index and target_digit are given, index takes priority.
        train_data_set (bool): Load from training (True) or test (False) set.
        show (bool): Display the image with matplotlib.

    Returns:
        tuple:
            - img_array (np.ndarray): Normalized image array, shape (28, 28), range [0, 1].
            - label (int): Digit label.
    """
    # Transform: PIL -> Tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(
        root="data",
        train=train_data_set,
        download=True,
        transform=transform
    )

    # ----------------------------------------
    # Case 1: Load by explicit index
    # ----------------------------------------
    if index is not None:
        img_tensor, label = mnist_dataset[index]

    # ----------------------------------------
    # Case 2: Load by target digit (0–9)
    # ----------------------------------------
    elif target_digit is not None:
        if not (0 <= target_digit <= 9):
            raise ValueError("target_digit must be an integer between 0 and 9.")

        # Search for the first example with this digit
        for i in range(len(mnist_dataset)):
            img_tensor, label = mnist_dataset[i]
            if label == target_digit:
                break
        else:
            raise ValueError(f"Digit {target_digit} not found in MNIST dataset!")

    else:
        raise ValueError("You must specify either `target_digit` or `index`.")

    # Convert to PIL
    img_pil = ToPILImage()(img_tensor)
    # Convert to numpy array
    img_array = img_tensor.squeeze().numpy()

    # Optional display
    if show:
        plt.imshow(img_pil, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

    # Save image
    # img_pil.save(f"mnist_{index}.png")

    return img_array, label


def get_emnist_image(csv_path):
    """
    Loads EMNIST CSV file (first column = label, next 784 columns = pixels).
    Returns images (N, 28, 28) and labels (N,)

    Returns the nth image (index) for a given target_label from EMNIST arrays.
    """
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    images_flat = df.iloc[:, 1:].values
    images = images_flat.reshape(-1, 28, 28)

    # Label 1 = 'A' in EMNIST Letters split
    target_label = 1
    # Get another handwritten 'A' (for example, the 5th 'A')
    index = 6  # nth_A - you can change this to 0, 1, 2, etc. to get different 'A's

    idx = np.where(labels == target_label)[0]
    if len(idx) == 0:
        raise ValueError(f"No examples found for label {target_label}")
    if index >= len(idx):
        raise IndexError(f"Only {len(idx)} examples available for label {target_label}")
    image_array = images[idx[index]]

    img = Image.fromarray(image_array.astype(np.uint8))
    # Rotate 90° clockwise
    rotated_img = img.transpose(Image.ROTATE_270)  # ROTATE_270 = rotate right

    rotated_img.save(f"emnist_letter_A_{index}.png")
    plt.figure(figsize=(2, 2))
    plt.imshow(rotated_img, cmap='gray')
    plt.axis('off')
    plt.show()


# # ============================================================
# # Digit Image Loading
# # ============================================================
# def load_digit_image(digit, size=(28, 28)):
#     """
#     Load a single digit as a numpy array scaled to [0,1].
#     Currently creates a placeholder image using PIL.
#     Replace with actual MNIST loader if needed.
#     """
#     img = Image.new("L", size, color=0)
#     draw = ImageDraw.Draw(img)
#     draw.text((4, 4), str(digit), fill=255)
#     img.save("test_img.png")
#     return np.array(img, dtype=np.float32) / 255.0


# ============================================================
# Image Perturbation Functions
# ============================================================
def apply_geometric_perturbation(img, perturb_type, params):
    """
    Apply geometric modifications to simulate 'what if' scenarios.
    Supported perturb_type values:
    - close_loop, break_loop, erase_line, add_line, connect_loops,
      extend_tail, straighten_arc, open_loop, dilate_stroke, remove_random_pixels
    """
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)

    if perturb_type == "close_loop":
        draw.line((5, 5, 10, 5), fill=255, width=params.get("width", 1))
    elif perturb_type == "break_loop":
        draw.line((5, 20, 10, 20), fill=0, width=params.get("width", 1))
    elif perturb_type == "erase_line":
        draw.rectangle((10, 10, 15, 15), fill=0)
    elif perturb_type == "add_line":
        draw.line((5, 14, 15, 14), fill=255, width=params.get("width", 1))
    elif perturb_type == "connect_loops":
        draw.line((5, 5, 15, 15), fill=255, width=params.get("width", 1))
    elif perturb_type == "extend_tail":
        draw.line((15, 20, 20, 23), fill=255, width=params.get("length", 1))
    elif perturb_type == "straighten_arc":
        draw.line((5, 5, 15, 5), fill=255, width=1)
    elif perturb_type == "open_loop":
        draw.rectangle((5, 15, 15, 20), fill=0)
    elif perturb_type == "dilate_stroke":
        img_pil = img_pil.filter(ImageFilter.MaxFilter(size=params.get("factor", 1)))
    elif perturb_type == "remove_random_pixels":
        img_arr = np.array(img_pil)
        h, w = img_arr.shape
        for _ in range(params.get("count", 5)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            img_arr[y, x] = 0
        img_pil = Image.fromarray(img_arr)

    return np.array(img_pil, dtype=np.float32) / 255.0


def flip_image(img):
    """Horizontally flip the image."""
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    flipped = ImageOps.mirror(img_pil)
    return np.array(flipped, dtype=np.float32) / 255.0


def adjust_brightness(img, factor):
    """Adjust image brightness by the given factor (>1 brighter, <1 darker)."""
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(factor)
    return np.array(img_pil, dtype=np.float32) / 255.0


def add_noise(img, amount=0.02):
    """Add random Gaussian noise to the image, clipped to [0,1]."""
    noise = np.random.randn(*img.shape) * amount
    noisy_img = np.clip(img + noise, 0.0, 1.0)
    return noisy_img


def blur_image(img, sigma=1.0):
    """Apply Gaussian blur to the entire image."""
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred, dtype=np.float32) / 255.0


# ============================================================
# Counterfactual / OOD
# ============================================================
def generate_counterfactual(img, model):
    """
    Placeholder for CF generator.
    For OOD input, returns None (safe).
    """
    return None


# ============================================================
# Counterfactual Metrics
# ============================================================
def compute_flip_rate(orig_pred, pert_pred):
    """Binary flip rate between two scalar predictions (0 or 1)."""
    return 1.0 if orig_pred != pert_pred else 0.0


def compute_proximity_delta(orig_img, pert_img):
    """Mean absolute pixel difference normalized to [0,1]."""
    return float(np.mean(np.abs(orig_img - pert_img)))


def compute_logit_stability(orig_logits, pert_logits):
    """Cosine similarity between original and perturbed logits."""
    orig = torch.tensor(orig_logits).float()
    pert = torch.tensor(pert_logits).float()
    cos_sim = F.cosine_similarity(orig, pert, dim=0)
    return float(cos_sim.item())


def compute_saliency(model, img):
    """Compute simple gradient-based saliency map for a single image."""
    model.eval()
    device = next(model.parameters()).device
    tensor = torch.tensor(img, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0).to(device)
    output = model(tensor)
    pred = output.argmax(dim=1)
    output[0, pred].backward()
    saliency = tensor.grad.abs().squeeze().cpu().numpy()
    saliency /= saliency.max() + 1e-8
    return saliency


if __name__ == "__main__":
    find_index_for_label(label_input=9)
    # get_mnist_image(index=80)

    # Update to your local Kaggle CSV path
    # train_csv = "data/EMNIST/raw/emnist-letters-train.csv"
    # get_emnist_image(train_csv)
