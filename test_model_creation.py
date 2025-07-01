#!/usr/bin/env python3
"""
Test layer creation and update method.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
from MultipoleNet import MultipoleNetRes, build_graph

dtype = np.float32

def test_model_creation():
    """Test that the model can be created and layers are set up correctly."""
    
    try:
        model = MultipoleNetRes(node_size=64, edge_size=64, num_steps=2)
        
        print(f"✅ Model created successfully")
        print(f"Number of mono layers: {len(model.gns_mono)}")
        print(f"Number of dipo layers: {len(model.gns_dipo)}")
        print(f"Number of quad layers: {len(model.gns_quad)}")
        print(f"Number of octu layers: {len(model.gns_octu)}")
        
        # Check that all layer lists have the same length
        assert len(model.gns_mono) == len(model.gns_dipo) == len(model.gns_quad) == len(model.gns_octu), "Layer count mismatch"
        
        print("✅ All layer counts match")
        
        # Test that update method can be called
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # Oxygen
            [0.757, 0.586, 0.0],   # Hydrogen 1
            [-0.757, 0.586, 0.0]   # Hydrogen 2
        ], dtype=dtype)
        
        elements = ['O', 'H', 'H']
        graph = build_graph(coordinates, elements)
        
        print("Testing update method...")
        result = model.update(graph)
        print(f"Update result: {type(result)}, length: {len(result) if result else None}")
        
        if result and len(result) == 4:
            print("✅ Update method works correctly")
            return True
        else:
            print("❌ Update method failed")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_creation()
