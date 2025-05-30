# SIK - Simplified Isolation Kernel

SIK is a Python library for text anomaly and outlier detection that implements Simplified Isolation Kernel. This package offers efficient implementations with support for sparse matrices, GPU acceleration, and various detection modes.

## Features

- **Multiple Implementation Options**:
  - `SIK`: Unified implementation with configurable backends
  - `BK_INNE`: Standard CPU implementation
  - `SparseINNE`: Memory-efficient implementation using sparse matrices
  - `GPU_BK_INNE`: Accelerated implementation using PyTorch and CUDA/MPS

- **Acceleration Support**:
  - CPU optimization
  - GPU acceleration via PyTorch (CUDA for NVIDIA GPUs)
  - Apple Metal Performance Shaders (MPS) for Apple Silicon

- **Detection Modes**:
  - Outlier detection (when training data may contain outliers)
  - Novelty detection (when training on normal samples only)

## Installation
install from source:

```bash
git clone https://github.com/charles-cao/SIK.git
```

### Requirements

- numpy>=2.2.6
- scikit_learn>=1.6.1
- scipy>=1.15.3
- torch>=2.5.1 (optional, required for GPU acceleration)

## Quick Start

```python
import numpy as np
from SIK import SIK

# Create some example embeddings (with outliers)
X_train = np.random.randn(1000, 10)
X_train[0:10] = X_train[0:10] + 5  # Add some outliers

# Create and train the model
model = SIK(
    max_samples=16,         # Number of samples per estimator
    n_estimators=200,       # Number of base estimators
    novelty=False,          # Outlier detection mode
    sparse=False,           # Use dense matrices
    device='auto',          # Auto-select best device
    random_state=42         # For reproducibility
)

model.fit(X_train)

# Get anomaly scores
scores = model.decision_function(X_train)

# Higher scores indicate higher likelihood of being anomalies
print("Top 10 potential outliers (indices):", np.argsort(scores)[-10:])
```

## API Reference

### SIK

```python
SIK(max_samples=16, n_estimators=200, novelty=False, 
               sparse=False, device='cpu', random_state=None)
```

Main class implementing the Boundary Kernel algorithm with flexible backend options.

**Parameters:**
- `max_samples` (int, default=16): Number of samples to draw for each estimator
- `n_estimators` (int, default=200): Number of base estimators in the ensemble
- `novelty` (bool, default=False): If True, for novelty detection (fit on normal data only)
- `sparse` (bool, default=False): If True, use sparse matrix implementation
- `device` (str, default='cpu'): Device to use - 'cpu', 'cuda', 'mps', or 'auto'
- `random_state` (int or RandomState, default=None): Random seed

**Methods:**
- `fit(data)`: Fit the model on training data
- `transform(X)`: Transform data into ensemble feature space
- `decision_function(X)`: Compute anomaly scores

### Legacy Classes

For backward compatibility, the following classes are also available:

- `BK_INNE`: Standard CPU implementation
- `SparseINNE`: Memory-efficient sparse matrix implementation
- `GPU_BK_INNE`: GPU-accelerated implementation

## Choosing the Right Implementation

- **Default usage**: `SIK` with default parameters
- **Memory constraints**: Set `sparse=True` for lower memory usage
- **Large datasets**: Use GPU acceleration with `device='auto'` or `device='cuda'`
- **Apple Silicon**: Set `device='mps'` for Metal acceleration
- **Compatibility**: Legacy classes for backward compatibility with existing code

## Algorithm Details

The Boundary Kernel method builds an ensemble of estimators where each estimator:
1. Samples a subset of points from the training data
2. Constructs hyperspheres around these points
3. Determines if test points fall inside or outside these hyperspheres
4. Aggregates results across estimators to identify anomalies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[GNU General Public License v3.0]
