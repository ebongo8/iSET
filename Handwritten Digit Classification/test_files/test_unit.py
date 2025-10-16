import torch
from test_files.utils import get_model, get_dataloaders, get_device_type


def test_model_io_shape():
    """
    TC-UT-01 Model I/O Shape Verification: Pass a random tensor of shape [1, 1, 28, 28] through the ImageClassifier model.
    Expected result: The output tensor shape is [1, 10].
    """
    # TODO Heather review/update code below
    model = get_model()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_dataloader_shapes():
    """
    TC-UT-02 Data Loader Verification: Instantiate the train_loader and retrieve one batch.
    Expected result: The batch contains two tensors: one of shape [32, 1, 28, 28] (images) and one of shape [32] (labels).
    """
    # TODO Heather review/update code below
    train_loader, _ = get_dataloaders()
    images, labels = next(iter(train_loader))
    assert images.shape == (32, 1, 28, 28)
    assert labels.shape == (32,)


def test_device_placement():
    """
    TC-UT-03 Device Placement Verification: Instantiate the model and move it to the target device.
    Expected result: next(model.parameters()).device returns the correct device (e.g., 'cuda:0' or 'cpu').
    """
    device_type = get_device_type()
    model = get_model()
    # Get the device of the first parameter of the model
    param_device = next(model.parameters()).device
    assert param_device.type == device_type


