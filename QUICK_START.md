# Quick Start Guide - Octupole Prediction

## Installation

```bash
pip install tensorflow numpy scipy graph-nets
```

## Basic Usage

### Load Model and Predict

```python
from MultipoleNet import load_model
import numpy as np

# Load the extended model
model = load_model()

# Define a water molecule
coordinates = np.array([
    [0.0, 0.0, 0.0],      # Oxygen
    [0.757, 0.586, 0.0],   # Hydrogen 1
    [-0.757, 0.586, 0.0]   # Hydrogen 2
], dtype=np.float32)

elements = ['O', 'H', 'H']

# Predict all multipole moments up to octupoles
monopoles, dipoles, quadrupoles, octupoles = model.predict(coordinates, elements)

print(f"Monopoles shape: {monopoles.shape}")      # (3, 1)
print(f"Dipoles shape: {dipoles.shape}")          # (3, 3)
print(f"Quadrupoles shape: {quadrupoles.shape}")  # (3, 3, 3)
print(f"Octupoles shape: {octupoles.shape}")      # (3, 3, 3, 3, 3)
```

## Understanding the Output

### Tensor Shapes
- **Monopoles**: `(n_atoms, 1)` - Atomic charges
- **Dipoles**: `(n_atoms, 3)` - Dipole moment vectors
- **Quadrupoles**: `(n_atoms, 3, 3)` - Quadrupole moment tensors
- **Octupoles**: `(n_atoms, 3, 3, 3, 3)` - Octupole moment tensors (NEW!)

### Physical Interpretation
- **Monopoles**: Represent the total charge on each atom
- **Dipoles**: Capture the charge separation within each atom
- **Quadrupoles**: Describe the charge distribution's second moments
- **Octupoles**: Capture fourth-order charge distribution effects

## Testing Equivariance

### Rotation Invariance Test
```python
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

# Generate a random rotation
rotation = R.random()
r_matrix = rotation.as_matrix().astype(np.float32)

# Rotate coordinates
rotated_coords = tf.matmul(coordinates, r_matrix.T)

# Predict for both original and rotated coordinates
original_result = model.predict(coordinates, elements)
rotated_result = model.predict(rotated_coords, elements)

monopoles_orig, dipoles_orig, quadrupoles_orig, octupoles_orig = original_result
monopoles_rot, dipoles_rot, quadrupoles_rot, octupoles_rot = rotated_result

# Test monopole invariance
print("Monopoles invariant:", np.allclose(monopoles_orig, monopoles_rot, atol=1e-5))

# Test dipole vector transformation
dipoles_transformed = tf.matmul(dipoles_orig, r_matrix.T)
print("Dipoles equivariant:", np.allclose(dipoles_transformed, dipoles_rot, atol=1e-5))

# Test quadrupole tensor transformation  
quadrupoles_transformed = tf.matmul(r_matrix, tf.matmul(quadrupoles_orig, r_matrix.T))
print("Quadrupoles equivariant:", np.allclose(quadrupoles_transformed, quadrupoles_rot, atol=1e-5))

# Test octupole 4th-order tensor transformation
octupoles_transformed = tf.einsum('im,jn,ko,lp,mnop->ijkl', 
                                  r_matrix, r_matrix, r_matrix, r_matrix, octupoles_orig)
print("Octupoles equivariant:", np.allclose(octupoles_transformed, octupoles_rot, atol=1e-5))
```

## Working with Larger Molecules

### Using RDKit for Molecular Input
```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Create molecule from SMILES
smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

# Generate 3D coordinates
AllChem.EmbedMolecule(mol)

# Extract coordinates and elements
coordinates = mol.GetConformer(0).GetPositions().astype(np.float32)
elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

# Predict multipoles
monopoles, dipoles, quadrupoles, octupoles = model.predict(coordinates, elements)

print(f"Aspirin has {len(elements)} atoms")
print(f"Octupole tensor components per atom: {octupoles.shape[-4:]} = {np.prod(octupoles.shape[-4:])} values")
```

## Memory Considerations

### Octupole Memory Usage
```python
n_atoms = len(elements)
octupole_components = n_atoms * 3**4  # 81 components per atom

print(f"Number of atoms: {n_atoms}")
print(f"Octupole components: {octupole_components}")
print(f"Memory estimate: {octupole_components * 4 / 1024:.1f} KB (float32)")
```

### For Large Molecules (>100 atoms)
- Consider processing in fragments
- Monitor memory usage
- Use float16 precision if needed

## Common Use Cases

### 1. Electrostatic Potential Calculation
```python
# Use monopoles for ESP calculation
def calculate_esp(monopoles, coordinates, grid_points):
    """Calculate electrostatic potential at grid points"""
    from scipy.spatial.distance import cdist
    
    distances = cdist(coordinates, grid_points)
    esp = np.sum(monopoles.numpy() / distances, axis=0)
    return esp * 1389.35  # Convert to kJ/mol
```

### 2. Molecular Similarity
```python
# Compare octupole fingerprints
def octupole_similarity(octupoles1, octupoles2):
    """Calculate similarity based on octupole moments"""
    flat1 = tf.reshape(octupoles1, [-1])
    flat2 = tf.reshape(octupoles2, [-1])
    
    # Cosine similarity
    dot_product = tf.reduce_sum(flat1 * flat2)
    norm1 = tf.norm(flat1)
    norm2 = tf.norm(flat2)
    
    return dot_product / (norm1 * norm2)
```

### 3. Property Prediction
```python
# Use multipoles as features for other properties
def multipole_features(monopoles, dipoles, quadrupoles, octupoles):
    """Extract scalar features from multipole moments"""
    features = []
    
    # Monopole statistics
    features.extend([tf.reduce_sum(monopoles), tf.reduce_mean(tf.abs(monopoles))])
    
    # Dipole magnitude
    dipole_mags = tf.norm(dipoles, axis=1)
    features.extend([tf.reduce_sum(dipole_mags), tf.reduce_max(dipole_mags)])
    
    # Quadrupole trace and determinant
    quad_traces = tf.linalg.trace(quadrupoles)
    features.extend([tf.reduce_sum(quad_traces), tf.reduce_mean(tf.abs(quad_traces))])
    
    # Octupole norm (simplified)
    octu_norms = tf.norm(tf.reshape(octupoles, [octupoles.shape[0], -1]), axis=1)
    features.extend([tf.reduce_sum(octu_norms), tf.reduce_max(octu_norms)])
    
    return tf.stack(features)
```

## Next Steps

1. **Explore the Notebooks**: Check out `how_to_use.ipynb` for detailed examples
2. **Read the API Documentation**: See `API_DOCUMENTATION.md` for complete reference
3. **Training**: Use `example_training.ipynb` to train your own models
4. **Integration**: Incorporate multipole predictions into your workflows

## Getting Help

- Check the API documentation for detailed function references
- Look at the test files for usage examples
- Review the notebooks for complete workflows
- See the changelog for migration guidance from earlier versions
