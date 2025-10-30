import torch
from src.classifier_model import ImageClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


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


def get_mnist_image(index=1, train_data_set=True, show=True):
    """
    Load a single MNIST sample and return a PIL image + label.
    This function is just a backup if we ever need to access and view images from the MNIST dataset.
    """
    # transform used by the dataset (converts PIL -> Tensor)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist_dataset = datasets.MNIST(
        root="data",
        train=train_data_set,
        download=True,
        transform=transform  # <-- actually using the variable here
    )

    img_tensor, label = mnist_dataset[index]  # tensor shape: [1,28,28], values in [0,1]
    img_pil = ToPILImage()(img_tensor)         # convert back to PIL for display

    if show:
        plt.imshow(img_pil, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

    return img_pil, label


if __name__ == "__main__":
    get_mnist_image()
