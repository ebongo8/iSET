import torch
import copy
from test_files.utils import get_model, get_dataloaders, get_device_type
from src.classifier_model import ImageClassifier


def test_training_loop_integrity():
    """
    TC-IT-01 Training Loop Integrity: Capture model weights before and after a single training step (optimizer.step()).
    Expected result: The weights after the step are different from the weights before the step.
    """
    # TODO Heather review/update code below

    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)
    model = get_model()
    train_loader, _ = get_dataloaders()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    before_weights = copy.deepcopy(next(model.parameters()).clone())
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    after_weights = next(model.parameters()).clone()

    assert not torch.equal(before_weights, after_weights), "Weights should update after training step"


def test_save_load_integrity(tmp_path):
    """
    TC-IT-02 Save/Load Integrity: Save a model's state_dict, load it into a new model instance, and compare the weights.

    Expected results: The weights of the new model are identical to the weights of the original model.
    """
    # TODO Erin review/update code below
    # TODO figure out if we need to get the trained model or not (maybe use load_trained_model function in utils)

    model = get_model()
    save_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), save_path)

    device_type = get_device_type()
    device = torch.device(device_type)
    new_model = ImageClassifier().to(device)
    new_model.load_state_dict(torch.load(save_path, map_location=device_type))

    new_model.load_state_dict(torch.load(save_path))

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Weights must match after load"
