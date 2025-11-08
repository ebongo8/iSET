import torch
import pandas as pd
from test_files.utils import get_model, get_dataloaders, get_device_type, load_trained_model
import csv


def test_training_loop_integrity():
    """
    TC-IT-01 Training Loop Integrity: Capture model weights before and after a single training step (optimizer.step()).
    Expected result: The weights after the step are different from the weights before the step.
    """
    device_type = get_device_type(windows_os=False)
    device = torch.device(device_type)
    model = get_model()
    train_loader, _ = get_dataloaders()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    # Capture weights before training step
    before_weights = next(model.parameters()).detach().clone()

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Capture weights after training step
    after_weights = next(model.parameters()).detach().clone()

    # Check if weights updated
    test_passed = not torch.equal(before_weights, after_weights)
    assert test_passed, "Weights should update after training step"

    # Save flattened weights to CSV with match column
    results_file = "TC-IT-01_training_loop_weights.csv"
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Before Weights", "After Weights", "Match"])

        # Flatten tensors
        before_flat = before_weights.flatten().cpu().numpy()
        after_flat = after_weights.flatten().cpu().numpy()

        # Write one row per weight
        for b, a in zip(before_flat, after_flat):
            writer.writerow([float(b), float(a), float(b) == float(a)])


def test_save_load_integrity(tmp_path):
    """
    TC-IT-02 Save/Load Integrity: Save a model's state_dict, load it into a new model instance, and compare the weights.

    Expected results: The weights of the new model are identical to the weights of the original model.
    """
    device_type = get_device_type()
    device = torch.device(device_type)

    # Step 1: Create the original model
    original_model = get_model().to(device)

    # Step 2: Save the model’s weights
    save_path = tmp_path / "model_state_test.pth"
    torch.save(original_model.state_dict(), save_path)

    # Step 3: Create a new (empty) model and load the saved weights
    new_model = get_model().to(device)
    new_model.load_state_dict(torch.load(save_path, map_location=device))

    # Step 4: Compare all weights between original and loaded models
    for p1, p2 in zip(original_model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Weights must match after save/load integrity test"

    # Step 5: Export some sample weights to CSV for visual confirmation
    csv_path = "TC-IT-02_model_weights_comparison.csv"
    rows = []
    for (name, orig_param), (_, loaded_param) in zip(original_model.state_dict().items(),
                                                    new_model.state_dict().items()):
        orig_vals = orig_param.flatten().cpu().numpy()
        new_vals = loaded_param.flatten().cpu().numpy()
        for i in range(min(5, len(orig_vals))):  # limit to 5 entries per layer
            rows.append({
                "Layer": name,
                "Index": i,
                "Original_Model_Weights": float(orig_vals[i]),
                "New_Model_Weights": float(new_vals[i]),
                "Match": float(orig_vals[i]) == float(new_vals[i])
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nWeights comparison saved to: {csv_path}")
