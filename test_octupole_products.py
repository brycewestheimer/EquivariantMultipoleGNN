#!/usr/bin/env python3
"""
Simple test for octupole product functions.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf

dtype = np.float32

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=dtype)])
def get_octupole_products(vectors):
    # Create 4th order tensor (octupole)
    v_expanded_1 = tf.expand_dims(vectors, axis=-1)  # [N, 3, 1]
    v_expanded_2 = tf.expand_dims(vectors, axis=-2)  # [N, 1, 3]
    v_expanded_3 = tf.expand_dims(vectors, axis=-1)  # [N, 3, 1]
    v_expanded_4 = tf.expand_dims(vectors, axis=-2)  # [N, 1, 3]
    
    # Compute outer product: v_i v_j v_k v_l
    outer_ij = v_expanded_1 * v_expanded_2  # [N, 3, 3]
    outer_kl = v_expanded_3 * v_expanded_4  # [N, 3, 3]
    
    # Expand dimensions for final outer product
    outer_ij = tf.expand_dims(tf.expand_dims(outer_ij, axis=-1), axis=-1)  # [N, 3, 3, 1, 1]
    outer_kl = tf.expand_dims(tf.expand_dims(outer_kl, axis=1), axis=1)    # [N, 1, 1, 3, 3]
    
    # Final 4th order tensor
    octupole_tensor = outer_ij * outer_kl  # [N, 3, 3, 3, 3]
    
    return octupole_tensor

def test_octupole_products():
    """Test octupole product computation."""
    
    # Simple test vectors
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=dtype)
    
    try:
        result = get_octupole_products(vectors)
        print(f"Octupole products shape: {result.shape}")
        print(f"Expected shape: (3, 3, 3, 3, 3)")
        
        # Should be (3, 3, 3, 3, 3)
        assert result.shape == (3, 3, 3, 3, 3), f"Wrong shape: {result.shape}"
        
        print("✅ Octupole products test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Octupole products test failed: {e}")
        return False

if __name__ == "__main__":
    test_octupole_products()
