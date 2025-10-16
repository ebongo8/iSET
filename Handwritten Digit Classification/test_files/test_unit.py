import torch
from test_files.utils import get_model, get_dataloaders, get_device_type


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


