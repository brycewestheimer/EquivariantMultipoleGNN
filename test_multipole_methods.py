#!/usr/bin/env python3
"""
Test individual multipole methods.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
from MultipoleNet import MultipoleNetRes, build_graph

dtype = np.float32

def test_multipole_methods():
    """Test that individual multipole methods work."""
    
    try:
        model = MultipoleNetRes(node_size=64, edge_size=64, num_steps=2)
        
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # Oxygen
            [0.757, 0.586, 0.0],   # Hydrogen 1
            [-0.757, 0.586, 0.0]   # Hydrogen 2
        ], dtype=dtype)
        
        elements = ['O', 'H', 'H']
        graph = build_graph(coordinates, elements)
        
        print("Testing update method...")
        graphs_mono, graphs_dipo, graphs_quad, graphs_octu = model.update(graph)
        print("✅ Update method successful")
        
        print("Testing monopoles method...")
        monopoles = model.monopoles(graphs_mono)
        print(f"Monopoles shape: {monopoles.shape}")
        
        print("Testing dipoles method...")
        vectors = tf.gather(coordinates, graph.senders) - tf.gather(coordinates, graph.receivers)
        dipoles = model.dipoles(graphs_dipo, graph.edges, vectors)
        print(f"Dipoles shape: {dipoles.shape}")
        
        print("Testing quadrupoles method...")
        quadrupoles = model.quadrupoles(graphs_quad, graph.edges, vectors)
        print(f"Quadrupoles shape: {quadrupoles.shape}")
        
        print("Testing octupoles method...")
        octupoles = model.octupoles(graphs_octu, graph.edges, vectors)
        print(f"Octupoles shape: {octupoles.shape}")
        
        print("✅ All individual methods work!")
        
        print("Testing full call method...")
        result = model.call(graph, coordinates)
        print(f"Call result: {type(result)}, length: {len(result) if result else None}")
        
        if result and len(result) == 4:
            mono, dipo, quad, octu = result
            print(f"Final result shapes:")
            print(f"  Monopoles: {mono.shape}")
            print(f"  Dipoles: {dipo.shape}")
            print(f"  Quadrupoles: {quad.shape}")
            print(f"  Octupoles: {octu.shape}")
            print("✅ Full call method works!")
            return True
        else:
            print("❌ Full call method failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multipole_methods()
