#!/usr/bin/env python3
"""
Simple direct test.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
from MultipoleNet import MultipoleNetRes, build_graph

dtype = np.float32

def simple_test():
    """Simple test of model call."""
    
    try:
        model = MultipoleNetRes(node_size=64, edge_size=64, num_steps=2)
        
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # Oxygen
            [0.757, 0.586, 0.0],   # Hydrogen 1
        ], dtype=dtype)
        
        elements = ['O', 'H']
        graph = build_graph(coordinates, elements)
        
        # Direct call to model
        result = model.call(graph, coordinates)
        print(f"Direct call result: {result}")
        
        if result is not None:
            mono, dipo, quad, octu = result
            print(f"Shapes: mono={mono.shape}, dipo={dipo.shape}, quad={quad.shape}, octu={octu.shape}")
            print("✅ Success!")
        else:
            print("❌ Result is None")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
