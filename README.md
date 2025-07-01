# Equivariant Graph Neural Network for Atomic Multipoles

## Description
Repository for the Model used in the publication ['Learning Atomic Multipoles: Prediction of the Electrostatic Potential with Equivariant Graph Neural Networks'](https://pubs.acs.org/doi/10.1021/acs.jctc.1c01021).
The model was trained on the following publicly available [Dataset](https://www.research-collection.ethz.ch/handle/20.500.11850/509052).

**Extended Version**: This repository has been enhanced to predict multipole moments up to **octupoles (4th-order)** in addition to the original monopoles, dipoles, and quadrupoles.

## Features

### Multipole Predictions
The model now predicts atomic multipole moments up to 4th order:
- **Monopoles**: Scalar charges `(n_atoms, 1)`
- **Dipoles**: Vector components `(n_atoms, 3)`  
- **Quadrupoles**: 2nd-order tensors `(n_atoms, 3, 3)`
- **Octupoles**: 4th-order tensors `(n_atoms, 3, 3, 3, 3)` ✨ **NEW**

### Key Properties
- **Equivariant**: All multipole predictions transform correctly under 3D rotations
- **Detraced**: Tensors automatically satisfy physical constraints
- **Scalable**: Efficient implementation using TensorFlow operations
- **Modular**: Easy to extend to higher-order multipoles

## Quick Start

```python
from MultipoleNet import load_model

# Load the extended model
model = load_model()

# Predict all multipole moments up to octupoles
monopoles, dipoles, quadrupoles, octupoles = model.predict(coordinates, elements)

print(f"Octupole tensor shape: {octupoles.shape}")  # (n_atoms, 3, 3, 3, 3)
```

## Tensor Shapes and Transformations

| Multipole Order | Shape | Transformation |
|----------------|-------|----------------|
| Monopoles (0th) | `(n_atoms, 1)` | Invariant |
| Dipoles (1st) | `(n_atoms, 3)` | `v' = R × v` |
| Quadrupoles (2nd) | `(n_atoms, 3, 3)` | `Q'_ij = R_ik × R_jl × Q_kl` |
| Octupoles (4th) | `(n_atoms, 3, 3, 3, 3)` | `O'_ijkl = R_im × R_jn × R_ko × R_lp × O_mnop` |

## How to Use
Check out the jupyter notebooks for exemplary use cases:
- `how_to_use.ipynb`: Basic usage and equivariance testing
- `example_training.ipynb`: Training pipeline with octupole support

## Requirements
- Tensorflow: 2.6.2 or higher
- Numpy: 1.21.2 or higher  
- Graph-Nets: 1.1.0 or higher
- Scipy: For molecular structure handling

## Installation

```bash
pip install tensorflow numpy scipy graph-nets
```

## Model Architecture

The extended architecture includes:
- Separate embedding layers for each multipole order
- Dedicated GNN processing paths for monopoles, dipoles, quadrupoles, and octupoles
- Specialized tensor operations for 4th-order octupole computation
- Automatic detracing to ensure physical constraints

## Citation

If you use this extended model in your research, please cite the original paper:

```bibtex
@article{multipole_gnn_2021,
  title={Learning Atomic Multipoles: Prediction of the Electrostatic Potential with Equivariant Graph Neural Networks},
  journal={Journal of Chemical Theory and Computation},
  year={2021},
  doi={10.1021/acs.jctc.1c01021}
}
```

## Extensions and Future Work

This implementation demonstrates how to extend equivariant GNNs to higher-order tensor predictions. The modular design allows for:
- Addition of hexadecapole moments (6th-order)
- Custom detracing schemes
- Alternative loss functions for tensor training
- Integration with other molecular property prediction tasks
