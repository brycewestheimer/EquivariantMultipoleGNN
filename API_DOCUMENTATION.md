# API Documentation - Extended Multipole Prediction

## Overview

This document describes the API for the extended EquivariantMultipoleGNN model that predicts atomic multipole moments up to octupoles (4th-order tensors).

## Core Classes and Functions

### MultipoleNetRes

Main model class for predicting atomic multipoles.

```python
class MultipoleNetRes(tf.keras.Model):
    def __init__(self, node_size=64, edge_size=64, activation=mila, num_steps=4)
```

#### Parameters:
- `node_size` (int): Size of node embeddings (default: 64)
- `edge_size` (int): Size of edge embeddings (default: 64) 
- `activation`: Activation function (default: mila)
- `num_steps` (int): Number of message passing steps (default: 4)

#### Methods:

##### `predict(coordinates, elements)`
Predicts multipole moments for a molecule.

**Parameters:**
- `coordinates` (array): Atomic coordinates with shape `(n_atoms, 3)`
- `elements` (list): List of element symbols, e.g., `['O', 'H', 'H']`

**Returns:**
- `monopoles` (Tensor): Shape `(n_atoms, 1)` - atomic charges
- `dipoles` (Tensor): Shape `(n_atoms, 3)` - dipole vectors  
- `quadrupoles` (Tensor): Shape `(n_atoms, 3, 3)` - quadrupole tensors
- `octupoles` (Tensor): Shape `(n_atoms, 3, 3, 3, 3)` - octupole tensors

**Example:**
```python
import numpy as np
from MultipoleNet import load_model

# Water molecule coordinates
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],   # H
    [-0.757, 0.586, 0.0]   # H
], dtype=np.float32)

elements = ['O', 'H', 'H']

model = load_model()
monopoles, dipoles, quadrupoles, octupoles = model.predict(coords, elements)

print(f"Monopoles: {monopoles.shape}")    # (3, 1)
print(f"Dipoles: {dipoles.shape}")        # (3, 3)  
print(f"Quadrupoles: {quadrupoles.shape}") # (3, 3, 3)
print(f"Octupoles: {octupoles.shape}")     # (3, 3, 3, 3, 3)
```

##### `call(graphs, coordinates)`
Lower-level prediction method using pre-built graphs.

**Parameters:**
- `graphs` (GraphsTuple): Pre-built graph representation
- `coordinates` (Tensor): Atomic coordinates

**Returns:** Same as `predict()` method

### Utility Functions

#### `load_model()`
Loads a pre-trained model with octupole support.

```python
from MultipoleNet import load_model

model = load_model()
```

**Returns:** Initialized `MultipoleNetRes` model with loaded weights

**Note:** Octupole weights are initialized randomly if not available in saved model.

#### `build_graph(coords, elements, cutoff=4.0, num_kernels=32)`
Builds a graph representation for a molecule.

**Parameters:**
- `coords` (array): Atomic coordinates `(n_atoms, 3)`
- `elements` (list): Element symbols
- `cutoff` (float): Distance cutoff for edges (default: 4.0 Å)
- `num_kernels` (int): Number of radial basis functions (default: 32)

**Returns:** `GraphsTuple` object for use with model

#### `build_graph_batched(coords, elements, cutoff=4.0, num_kernels=32)`
Builds batched graphs for multiple conformations.

**Parameters:**
- `coords` (array): Batch of coordinates `(n_batch, n_atoms, 3)`
- `elements` (list): Element symbols
- `cutoff` (float): Distance cutoff for edges
- `num_kernels` (int): Number of radial basis functions

**Returns:** Batched `GraphsTuple` for efficient processing

### Tensor Operations

#### `get_octupole_products(vectors)`
Computes 4th-order outer products for octupole calculation.

**Parameters:**
- `vectors` (Tensor): Distance vectors `(n_edges, 3)`

**Returns:** 4th-order tensor `(n_edges, 3, 3, 3, 3)`

#### `D_O(octupoles)`
Detraces 4th-order octupole tensors.

**Parameters:**
- `octupoles` (Tensor): Raw octupole tensors `(n_atoms, 3, 3, 3, 3)`

**Returns:** Detraced octupole tensors with same shape

#### `D_Q(quadrupoles)`
Detraces 2nd-order quadrupole tensors.

**Parameters:**
- `quadrupoles` (Tensor): Raw quadrupole tensors `(n_atoms, 3, 3)`

**Returns:** Detraced quadrupole tensors with same shape

## Equivariance Properties

All multipole predictions are equivariant under 3D rotations:

### Monopoles (Invariant)
```python
# Monopoles remain unchanged under rotation
monopoles_original = model.predict(coords, elements)[0]
monopoles_rotated = model.predict(rotated_coords, elements)[0]
# monopoles_original ≈ monopoles_rotated
```

### Dipoles (Vector Transformation)
```python
# Dipoles transform as vectors: v' = R @ v
dipoles_original = model.predict(coords, elements)[1]
dipoles_rotated = model.predict(rotated_coords, elements)[1]
dipoles_transformed = tf.matmul(dipoles_original, R.T)
# dipoles_rotated ≈ dipoles_transformed
```

### Quadrupoles (2nd-order Tensor)
```python
# Quadrupoles: Q'_ij = R_ik R_jl Q_kl
quadrupoles_original = model.predict(coords, elements)[2]
quadrupoles_rotated = model.predict(rotated_coords, elements)[2]
quadrupoles_transformed = tf.matmul(R, tf.matmul(quadrupoles_original, R.T))
# quadrupoles_rotated ≈ quadrupoles_transformed
```

### Octupoles (4th-order Tensor)
```python
# Octupoles: O'_ijkl = R_im R_jn R_ko R_lp O_mnop
octupoles_original = model.predict(coords, elements)[3]
octupoles_rotated = model.predict(rotated_coords, elements)[3]
octupoles_transformed = tf.einsum('im,jn,ko,lp,mnop->ijkl', 
                                  R, R, R, R, octupoles_original)
# octupoles_rotated ≈ octupoles_transformed
```

## Training Interface

### Loss Functions

#### Monopole Loss
```python
def get_loss_mono(monopoles_predicted, monopoles_ref):
    return tf.reduce_mean(tf.math.squared_difference(monopoles_predicted, monopoles_ref))
```

#### Dipole Loss
```python
def get_loss_dipo(dipoles_predicted, dipoles_ref):
    return tf.reduce_mean(tf.math.squared_difference(dipoles_predicted, dipoles_ref))
```

#### Quadrupole Loss (Upper Triangle Only)
```python
mask = tf.linalg.band_part(tf.ones((3, 3), dtype=tf.bool), 0, -1)
def get_loss_quad(quadrupoles_predicted, quadrupoles_ref):
    return tf.reduce_mean(tf.boolean_mask(
        tf.math.squared_difference(quadrupoles_predicted, quadrupoles_ref), 
        mask, axis=1))
```

#### Octupole Loss (All Components)
```python
def get_loss_octu(octupoles_predicted, octupoles_ref):
    return tf.reduce_mean(tf.math.squared_difference(octupoles_predicted, octupoles_ref))
```

### Variable Collections

Functions to collect trainable variables for each multipole order:

```python
# Get variables for specific multipole training
mono_vars = variables_mono(model)    # Monopole variables
dipo_vars = variables_dipo(model)    # Dipole variables  
quad_vars = variables_quad(model)    # Quadrupole variables
octu_vars = variables_octu(model)    # Octupole variables
```

## Data Format

### TFRecord Features
For training data storage:

```python
feature_description = {
    'nodes': tf.io.FixedLenFeature([], tf.string),
    'edges': tf.io.FixedLenFeature([], tf.string),
    'coordinates': tf.io.FixedLenFeature([], tf.string),
    'n_node': tf.io.FixedLenFeature([], tf.string),
    'n_edge': tf.io.FixedLenFeature([], tf.string),
    'senders': tf.io.FixedLenFeature([], tf.string),
    'receivers': tf.io.FixedLenFeature([], tf.string),
    'monopoles': tf.io.FixedLenFeature([], tf.string),
    'dipoles': tf.io.FixedLenFeature([], tf.string),
    'quadrupoles': tf.io.FixedLenFeature([], tf.string),
    'octupoles': tf.io.FixedLenFeature([], tf.string),  # NEW
}
```

## Memory and Performance Considerations

### Tensor Sizes
- **Monopoles**: 1 component per atom
- **Dipoles**: 3 components per atom  
- **Quadrupoles**: 9 components per atom (stored as 3×3 matrix)
- **Octupoles**: 81 components per atom (stored as 3×3×3×3 tensor)

### Memory Usage
For a molecule with N atoms:
- **Total components**: N × (1 + 3 + 9 + 81) = N × 94 components
- **Memory scaling**: O(N) for molecules, O(N⁴) for octupole tensor operations

### Performance Tips
1. Use batched predictions for multiple conformations
2. Consider lower precision (float16) for large molecules
3. Octupole predictions are computationally expensive - profile memory usage
4. For inference-only applications, consider freezing lower-order predictions

## Error Handling

Common issues and solutions:

### Shape Mismatches
```python
# Ensure coordinates are float32
coords = coords.astype(np.float32)

# Check element symbols are valid
valid_elements = ['H', 'C', 'N', 'O', 'F', 'S', 'CL', 'Cl']
assert all(elem in valid_elements for elem in elements)
```

### Memory Issues
```python
# For large molecules, process in smaller batches
if len(elements) > 100:
    # Consider splitting into smaller fragments
    pass
```

### Missing Octupole Weights
If octupole weights are not available in a saved model:
```python
# The load_model() function automatically handles this
# Octupole layers will be initialized with random weights
model = load_model()  # Safe even without octupole weights
```
