import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from classifier_model import ImageClassifier


def get_device(windows_os=False):
    if windows_os:
        # return "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_and_save_model(save_path="model_state.pt", epochs=10):
    # Run this function once and save the model in the model_state.pt file to load for later

    # Prepare data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create an instance of the image classifier model
    device = get_device()
    model = ImageClassifier().to(device)

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() # accumulate loss across batches

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_and_save_model(save_path="model_state.pt", epochs=10)
