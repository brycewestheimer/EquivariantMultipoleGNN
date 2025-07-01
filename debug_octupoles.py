#!/usr/bin/env python3
"""
Debug octupole shapes.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import tensorflow as tf
from MultipoleNet import MultipoleNetRes, build_graph, get_octupole_products

dtype = np.float32

def debug_octupole_shapes():
    """Debug the shapes in octupole computation."""
    
    try:
        model = MultipoleNetRes(node_size=64, edge_size=64, num_steps=2)
        
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # Oxygen
            [0.757, 0.586, 0.0],   # Hydrogen 1
            [-0.757, 0.586, 0.0]   # Hydrogen 2
        ], dtype=dtype)
        
        elements = ['O', 'H', 'H']
        graph = build_graph(coordinates, elements)
        
        print(f"Graph info:")
        print(f"  Nodes: {graph.nodes.shape}")
        print(f"  Edges: {graph.edges.shape}")
        print(f"  Senders: {graph.senders.shape}")
        print(f"  Receivers: {graph.receivers.shape}")
        print(f"  n_node: {graph.n_node}")
        print(f"  n_edge: {graph.n_edge}")
        
        graphs_mono, graphs_dipo, graphs_quad, graphs_octu = model.update(graph)
        
        vectors = tf.gather(coordinates, graph.senders) - tf.gather(coordinates, graph.receivers)
        print(f"Vectors shape: {vectors.shape}")
        
        octu_products = get_octupole_products(vectors)
        print(f"Octupole products shape: {octu_products.shape}")
        
        octu_features_senders = tf.gather(graphs_octu.nodes, graphs_octu.senders)
        octu_features_receivers = tf.gather(graphs_octu.nodes, graphs_octu.receivers)
        features_octu = tf.concat((octu_features_senders, octu_features_receivers, graph.edges), axis=-1)
        print(f"Octupole features shape: {features_octu.shape}")
        
        weights = model.octu(features_octu)
        print(f"Weights shape: {weights.shape}")
        
        weights_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(weights, axis=-1), axis=-1), axis=-1)
        print(f"Weights expanded shape: {weights_expanded.shape}")
        
        weighted_octu_products = octu_products * weights_expanded
        print(f"Weighted octupole products shape: {weighted_octu_products.shape}")
        
        print(f"Receivers for aggregation: {graph.receivers.numpy()}")
        print(f"Number of segments: {tf.reduce_sum(graph.n_node).numpy()}")
        
        result = tf.math.unsorted_segment_sum(weighted_octu_products, graph.receivers, num_segments=tf.reduce_sum(graph.n_node))
        print(f"Final octupole result shape: {result.shape}")
        
        return True
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_octupole_shapes()
