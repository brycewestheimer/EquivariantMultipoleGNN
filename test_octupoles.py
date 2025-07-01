#!/usr/bin/env python3
"""
Test script to verify that the octupole extension works correctly.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
from MultipoleNet import MultipoleNetRes, build_graph

dtype = np.float32

def test_octupole_functionality():
    """Test that the model can handle octupoles without errors."""
    
    # Create a simple model for testing
    model = MultipoleNetRes(node_size=64, edge_size=64, num_steps=2)
    
    # Create a simple test molecule (water)
    coordinates = np.array([
        [0.0, 0.0, 0.0],      # Oxygen
        [0.757, 0.586, 0.0],   # Hydrogen 1
        [-0.757, 0.586, 0.0]   # Hydrogen 2
    ], dtype=dtype)
    
    elements = ['O', 'H', 'H']
    
    # Build graph
    graph = build_graph(coordinates, elements)
    
    # Test prediction
    try:
        print("Building graph...")
        graph = build_graph(coordinates, elements)
        print(f"Graph created successfully. Nodes: {graph.nodes.shape if graph.nodes is not None else None}")
        print(f"Graph edges: {graph.edges.shape if graph.edges is not None else None}")
        print(f"Graph senders: {graph.senders.shape if graph.senders is not None else None}")
        print(f"Graph receivers: {graph.receivers.shape if graph.receivers is not None else None}")
        
        print("Testing model call directly...")
        result = model(graph, coordinates)
        print(f"Direct call result type: {type(result)}")
        print(f"Direct call result: {result}")
        
        if result is not None:
            monopoles, dipoles, quadrupoles, octupoles = result
            
            print("Octupole extension successful!")
            print(f"Monopoles shape: {monopoles.shape}")
            print(f"Dipoles shape: {dipoles.shape}")
            print(f"Quadrupoles shape: {quadrupoles.shape}")
            print(f"Octupoles shape: {octupoles.shape}")
            
            # Check that octupoles have the correct shape (N_atoms, 3, 3, 3, 3)
            expected_octu_shape = (len(elements), 3, 3, 3, 3)
            assert octupoles.shape == expected_octu_shape, f"Expected octupole shape {expected_octu_shape}, got {octupoles.shape}"
            
            print("All shape checks passed!")
            return True
        else:
            print("Model call returned None")
            return False
        
    except Exception as e:
        print(f"Error during octupole prediction: {e}")
        return False

if __name__ == "__main__":
    print("Testing octupole functionality...")
    success = test_octupole_functionality()
    
    if success:
        print("\n✅ Octupole extension test passed!")
    else:
        print("\n❌ Octupole extension test failed!")
