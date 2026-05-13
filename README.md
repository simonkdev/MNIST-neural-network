# MNIST Neural Network

A **neural network implementation from scratch** for the **MNIST handwritten digit recognition dataset**, built using only **NumPy**. This educational project demonstrates the core concepts of neural networks without relying on machine learning frameworks like TensorFlow or PyTorch.

## 📍 About

### What This Project Does
This project implements a **feedforward neural network** that:
- Takes 28×28 grayscale images of handwritten digits (0-9) as input
- Processes them through multiple hidden layers
- Outputs a probability distribution over the 10 possible digit classes
- Achieves digit classification through forward and backward propagation

### Purpose
This project was built **for practice and understanding** of neural network fundamentals, including:
- Matrix operations in neural networks
- Forward and backward propagation
- Activation functions (Sigmoid, ReLU)
- Gradient descent and weight updates
- Cost functions and optimization

### Educational Focus
✅ **No ML frameworks** – Built from scratch using only NumPy
✅ **Transparent implementation** – Every mathematical operation is explicit
✅ **Multiple versions** – Evolution from basic to improved architectures
✅ **Practical example** – Works with real MNIST dataset

### Target Audience
- Students learning about neural networks
- Developers wanting to understand ML fundamentals
- Anyone interested in implementing neural networks from first principles

## ✨ Features

### Core Implementations
| Version | File | Activation | Output Layer | Status |
|---------|------|------------|--------------|--------|
| Original | `nnw.ipynb` | Sigmoid | Sigmoid | Initial implementation |
| Fixed | `nnw_fixed.ipynb` | Sigmoid | Sigmoid | Bug fixes applied |
| v3 | `v3.ipynb` | ReLU | Softmax | Improved architecture |
| Standalone | `v3.py` | ReLU | Softmax | Python script version |

### Neural Network Components
- **Forward Propagation**: Multi-layer computation with configurable architectures
- **Backward Propagation**: Gradient calculation for all layers
- **Activation Functions**:
  - Sigmoid (original versions)
  - ReLU with derivative (v3)
  - Softmax for output layer (v3)
- **Cost Function**: Mean squared error / Cross-entropy
- **Optimization**: Gradient descent with configurable learning rate

### Data Handling
- **MNIST Dataset**: Full dataset loading (60,000 training, 10,000 test images)
- **Preprocessing**:
  - Image flattening (28×28 → 784 features)
  - Bias term addition
  - Feature normalization (mean subtraction, std division)
  - One-hot encoding for labels
- **Image Utilities**: PNG to binary array conversion for custom inputs

### Implementation Details
- **Pure NumPy**: No ML frameworks, only NumPy for matrix operations
- **Modular Design**: Separate functions for each neural network component
- **Configurable**: Learning rate, iterations, layer sizes are all parameterized
- **Reproducible**: Fixed random seeds where applicable

## 🎠 Architecture

### Neural Network Structure

#### Version 1 & 2 (nnw.ipynb, nnw_fixed.ipynb)
```
Input Layer (785 neurons: 784 pixels + 1 bias)
    ↓ (Sigmoid)
Hidden Layer (3 neurons)
    ↓ (Sigmoid)
Output Layer (1 neuron)
```

#### Version 3 (v3.ipynb, v3.py)
```
Input Layer (785 neurons: 784 pixels + 1 bias)
    ↓ (ReLU)
Hidden Layer 1 (64 neurons)
    ↓ (ReLU)
Hidden Layer 2 (128 neurons)
    ↓ (ReLU)
Output Layer (10 neurons: digits 0-9)
    ↓ (Softmax)
```

### Key Components

#### Activation Functions
| Function | Version | Formula | Derivative |
|----------|---------|---------|------------|
| Sigmoid | v1, v2 | 1 / (1 + e^(-x)) | σ(x) × (1 - σ(x)) |
| ReLU | v3 | max(0, x) | 1 if x > 0, else 0 |
| Softmax | v3 | e^x / ∑e^x | Jacobian matrix |

#### Forward Propagation
```python
# v3 implementation
z1 = X.dot(theta_1)           # Weighted sum
a1 = relu(z1)                # Activation
a1 = np.hstack((a1, ones))    # Add bias

z2 = a1.dot(theta_2)          # Weighted sum
a2 = relu(z2)                # Activation
a2 = np.hstack((a2, ones))    # Add bias

z3 = a2.dot(theta_out)       # Weighted sum
y_hat = softmax(z3)          # Final output
```

#### Backpropagation
- Computes gradients for each layer
- Updates weights using gradient descent
- Learning rate: 0.007 (v3)
- Iterations: 100 (v3)

### Data Flow
```
MNIST Images (28×28)
    ↓
Flatten + Normalize (784 features + 1 bias)
    ↓
Forward Pass (through all layers)
    ↓
Softmax Output (10 probabilities)
    ↓
Compare with True Labels
    ↓
Backpropagate Error
    ↓
Update Weights
```

### Implementation Evolution
| Aspect | v1/v2 | v3 |
|--------|-------|----|
| Activation | Sigmoid | ReLU |
| Output | Single neuron | 10 neurons (one per digit) |
| Output Activation | Sigmoid | Softmax |
| Hidden Layers | 1 layer (3 neurons) | 2 layers (64 + 128 neurons) |
| Performance | Basic | Improved accuracy |

## 🚀 Usage

### Prerequisites
- **Python 3.13+** (recommended)
- **Devenv** (for environment management)
- **Direnv** (optional, for automatic environment activation)

### Setup

#### Using Devenv
```sh
# Enter the development environment
devenv shell

# Or use direnv (if .envrc is configured)
cd MNIST-neural-network
```

The Devenv environment includes:
- NumPy
- Pillow (PIL)
- pandas
- seaborn
- matplotlib
- Jupyter

#### Manual Setup
```sh
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venvScriptsactivate

# Install dependencies
pip install numpy Pillow pandas seaborn matplotlib jupyter
```

### Running the Neural Network

#### Jupyter Notebooks
```sh
# Start Jupyter
jupyter notebook

# Then open one of:
# - nnw.ipynb (original)
# - nnw_fixed.ipynb (fixed version)
# - v3.ipynb (improved version)
```

#### Standalone Script (v3.py)
```sh
# Run the v3 implementation
python v3.py

# This will:
# 1. Load MNIST data
# 2. Initialize weights
# 3. Train the network
# 4. Display sample predictions
```

### Testing with Custom Images
The notebooks include a `png_to_binary_array()` function that converts 28×28 PNG images to the format expected by the network:

```python
from PIL import Image
import numpy as np

def png_to_binary_array(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    img_array = np.append(img_array, 1)  # Add bias term
    return img_array.flatten()
```

### Expected Output
After training, the network will output:
- Training progress (per iteration)
- Final accuracy on test set
- Sample predictions with confidence scores

## 📁 Project Structure

```
MNIST-neural-network/
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── .envrc                 # Direnv configuration for Devenv
├── .gitignore             # Git ignore patterns
└── data/                  # MNIST dataset files
    ├── x_train.npy        # Training images (60,000 samples)
    ├── y_train.npy        # Training labels
    ├── x_test.npy         # Test images (10,000 samples)
    └── y_test.npy         # Test labels
├── devenv.nix             # Devenv package definitions
├── devenv.yaml            # Devenv configuration
├── devenv.lock            # Locked dependency versions
├── nnw.ipynb              # Original implementation (sigmoid)
├── nnw_fixed.ipynb         # Fixed version (sigmoid)
├── v3.ipynb               # Improved version (ReLU + softmax)
└── v3.py                  # Standalone script (ReLU + softmax)
```

### File Descriptions

| File | Purpose | Key Features |
|------|---------|--------------|
| `nnw.ipynb` | Original neural network | Sigmoid activation, single output neuron |
| `nnw_fixed.ipynb` | Fixed implementation | Corrected bugs from original |
| `v3.ipynb` | Version 3 | ReLU activation, softmax output, 2 hidden layers |
| `v3.py` | Standalone script | Same as v3.ipynb but runs without Jupyter |
| `data/*.npy` | MNIST dataset | Pre-loaded NumPy arrays |
| `devenv.nix` | Environment | Defines Python packages |
| `devenv.yaml` | Devenv config | Specifies inputs and options |

## 💻 Development Environment

This project uses **[Devenv](https://devenv.sh/)** for reproducible development environments.

### Devenv Configuration

#### `devenv.yaml`
```yaml
inputs:
  nixpkgs:
    url: github:cachix/devenv-nixpkgs/rolling
```

Uses the rolling release of devenv-nixpkgs for up-to-date Python packages.

#### `devenv.nix`
```nix
{
  pkgs,
  lib,
  config,
  ...
}: {
  packages = with pkgs; [
    python313Packages.numpy
  ];

  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.venv.requirements = ''
    Pillow
    pandas
    seaborn
    matplotlib
    jupyter
  '';
}
```

### Features

#### Automatic Environment Activation
The `.envrc` file enables **Direnv** integration:
```sh
# With direnv installed:
cd MNIST-neural-network
# Environment automatically activates
```

#### Available Packages
| Package | Purpose |
|---------|---------|
| NumPy | Numerical computing |
| Pillow | Image processing |
| pandas | Data manipulation |
| seaborn | Statistical visualization |
| matplotlib | Plotting library |
| Jupyter | Interactive notebooks |

### Devenv Commands

| Command | Description |
|---------|-------------|
| `devenv shell` | Enter the development environment |
| `devenv test` | Test the environment |
| `devenv search <package>` | Search for packages |
| `devenv tasks list` | List available tasks |

### Benefits
✅ **Reproducible**: Same environment across all systems
✅ **Isolated**: No conflicts with system Python
✅ **Version-controlled**: Dependencies tracked in `devenv.lock`
✅ **Easy setup**: Single command to get started

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full license text.

Copyright (c) 2026 Simon Korten
