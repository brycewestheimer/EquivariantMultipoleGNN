# Changelog

All notable changes to the EquivariantMultipoleGNN project will be documented in this file.

## [2.0.0] - 2025-07-01

### Added - Octupole Support 🎉

#### Core Model Extensions
- **Octupole Prediction**: Extended model to predict 4th-order multipole moments
  - New `embedding_octu` layer for octupole-specific node embeddings
  - New `gns_octu` layers for octupole graph neural network processing  
  - New `octu` output head for octupole moment prediction
  - Model now returns 4-tuple: `(monopoles, dipoles, quadrupoles, octupoles)`

#### Tensor Operations
- **4th-order Tensor Support**: New functions for octupole computation
  - `get_octupole_products()`: Computes 4th-order outer products from distance vectors
  - `D_O()`: Detracing function for 4th-order tensors to ensure physical constraints
  - Proper broadcasting for 5-dimensional tensor operations

#### Training Infrastructure
- **Extended Training Pipeline**: Full support for octupole training
  - New loss function `get_loss_octu()` for octupole moment prediction
  - New variable collection function `variables_octu()` for octupole-specific parameters
  - Updated training functions to handle 4-tuple outputs
  - Support for separate optimizers per multipole order

#### Documentation
- **Comprehensive API Documentation**: Complete documentation for all new features
  - Detailed function signatures and usage examples
  - Equivariance property explanations for 4th-order tensors
  - Memory and performance considerations
  - Training pipeline documentation

#### Notebooks
- **Updated Examples**: All example notebooks updated for octupole support
  - `how_to_use.ipynb`: Demonstrates octupole prediction and equivariance testing
  - `example_training.ipynb`: Shows complete training pipeline with octupoles
  - Added octupole rotation tests using 4th-order tensor transformations

### Changed

#### Model Interface
- **Breaking Change**: `predict()` method now returns 4-tuple instead of 3-tuple
  ```python
  # Before (v1.x)
  monopoles, dipoles, quadrupoles = model.predict(coords, elements)
  
  # After (v2.0)  
  monopoles, dipoles, quadrupoles, octupoles = model.predict(coords, elements)
  ```

#### Tensor Shapes
- **New Output Shape**: Octupoles have shape `(n_atoms, 3, 3, 3, 3)`
- **Memory Impact**: Increased memory usage due to 4th-order tensors (81 components per atom)

#### Data Format
- **TFRecord Updates**: Training data format now includes octupole fields
  - Added `'octupoles'` field to feature descriptions
  - Updated data loading functions to parse octupole tensors
  - Automatic detracing applied during data loading

### Technical Details

#### Equivariance Properties
- **4th-order Tensor Rotation**: Octupoles transform as `O'_ijkl = R_im R_jn R_ko R_lp O_mnop`
- **Proper Detracing**: Removes traces over index pairs (i,j) and (k,l)
- **Physical Constraints**: Ensures octupole tensors satisfy mathematical requirements

#### Performance Optimizations
- **Efficient Tensor Operations**: Uses TensorFlow's einsum for optimal performance
- **Memory Management**: Careful dimension handling to avoid unnecessary memory allocation
- **Batch Processing**: Support for batched octupole computation

#### Backward Compatibility
- **Model Loading**: Gracefully handles models without octupole weights
- **Progressive Enhancement**: Existing monopole/dipole/quadrupole functionality unchanged
- **Safe Fallbacks**: Missing octupole weights are initialized randomly

### Testing

#### New Test Suite
- **Comprehensive Testing**: Full test coverage for octupole functionality
  - `test_octupoles.py`: End-to-end octupole prediction testing
  - `test_octupole_products.py`: 4th-order tensor operation validation
  - `test_model_creation.py`: Architecture consistency checks
  - Shape validation and equivariance property testing

#### Validation Results
- ✅ Correct tensor shapes for all multipole orders
- ✅ Proper equivariance under 3D rotations  
- ✅ Numerical stability of 4th-order operations
- ✅ Memory efficiency and computational performance
- ✅ Integration with existing codebase

### Migration Guide

#### For Users
1. **Update prediction calls**:
   ```python
   # Add octupoles to unpacking
   monopoles, dipoles, quadrupoles, octupoles = model.predict(coords, elements)
   ```

2. **Handle new tensor shape**:
   ```python
   print(f"Octupole shape: {octupoles.shape}")  # (n_atoms, 3, 3, 3, 3)
   ```

#### For Developers
1. **Update training loops** to handle 4-tuple returns
2. **Add octupole loss terms** to total loss computation
3. **Include octupole variables** in optimizer configurations
4. **Update data pipelines** to include octupole ground truth

### Dependencies

#### Requirements
- TensorFlow >= 2.6.2 (no change)
- NumPy >= 1.21.2 (no change)
- Graph-Nets >= 1.1.0 (no change)
- SciPy (for molecular structure handling)

#### Installation
```bash
pip install tensorflow numpy scipy graph-nets
```

### Known Issues
- **Memory Usage**: Octupole tensors require significant memory for large molecules
- **Training Time**: 4th-order tensor operations increase computational cost
- **Weight Compatibility**: Models trained on v1.x cannot directly load octupole weights

### Future Roadmap
- [ ] Hexadecapole moments (6th-order tensors)
- [ ] Optimized memory layouts for high-order tensors
- [ ] Alternative detracing schemes
- [ ] Integration with other molecular property predictions
- [ ] GPU-optimized tensor operations

---

## [1.0.0] - 2021

### Initial Release
- Monopole, dipole, and quadrupole prediction
- Equivariant graph neural networks
- Training pipeline and example notebooks
- Pre-trained model weights

---

**Note**: This project follows [Semantic Versioning](https://semver.org/). The addition of octupole support represents a major version increment (1.x → 2.0) due to the breaking change in the API return format.
