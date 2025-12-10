# MNIST Handwritten Digit Classification with PyTorch

This project demonstrates a simple implementation of a deep learning model for classifying handwritten digits from the **MNIST dataset** using the **PyTorch** library. The MNIST dataset is a widely-used benchmark in computer vision [1].

## Project Overview

This project uses an **open-source supervised learning model for MNIST handwritten digit classification [1]** as the model to be tested. While the model itself classifies handwritten digits using a Convolutional Neural Network (CNN) built with PyTorch and trained with the Adam optimizer, the primary goal of this repository is to **develop a modular and extensible framework for testing deep neural networks (DNNs)**.

The project involves the following steps:

* Loading and preprocessing the MNIST dataset
* Designing and building a CNN model architecture
* Training the model on the training data
* Evaluating the model's performance on the test data
* Saving and loading the trained model
* Performing inference on new images

The code has been modularized for clarity and testing:

```
src/
├── classifier_model.py      # ImageClassifier class
├── train_model.py           # Trains and saves model
test_files/
├── test_unit.py             # Unit test functions
├── test_integration.py      # Integration test functions
├── test_system.py           # System test functions
├── test_uat.py              # User acceptance test functions
├── test_cf.py               # Counterfactual reasoning test functions
├── test_xai.py              # XAI test functions
├── utils.py                 # Helper functions used across tests
test_images/                 # Images used in the tests
test_results/                # Output figures from the tests
```

## Requirements

* Python (3.x)
* PyTorch (1.x)
* torchvision
* PIL

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/mnist-classification-pytorch.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Testing

This project includes a **comprehensive test suite** using **Pytest**:

* **Unit tests:** Verify individual components (data loaders, model I/O, device selection)
* **Integration tests:** Validate interactions between modules
* **System tests:** End-to-end evaluation of training and inference
* **User Acceptance Tests (UAT):** Ensure model meets expected requirements
* **Counterfactual reasoning tests:** Assess model robustness to hypothetical changes
* **XAI tests:** Validate explainability and feature attribution methods

Run all tests with:

```
pytest test_files/
```

## References

[1] RafayKhattak, “GitHub - RafayKhattak/Digit-Classification-Pytorch: Simple MNIST Handwritten Digit Classification using Pytorch,” GitHub, 2025. https://github.com/RafayKhattak/Digit-Classification-Pytorch (accessed Dec. 02, 2025).
