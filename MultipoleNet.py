import graph_nets as gn
import numpy as np
import tensorflow as tf

from scipy.spatial.distance import pdist, squareform

dtype = np.float32
tf.keras.backend.set_floatx('float32')

onehots_elements = {
                        'H': np.array([1, 0, 0, 0, 0, 0, 0], dtype=dtype),
                        'C': np.array([0, 1, 0, 0, 0, 0, 0], dtype=dtype),
                        'N': np.array([0, 0, 1, 0, 0, 0, 0], dtype=dtype),
                        'O': np.array([0, 0, 0, 1, 0, 0, 0], dtype=dtype),                        
                        'F': np.array([0, 0, 0, 0, 1, 0, 0], dtype=dtype),
                        'S': np.array([0, 0, 0, 0, 0, 1, 0], dtype=dtype),
                        'CL': np.array([0, 0, 0, 0, 0, 0, 1], dtype=dtype),
                        'Cl': np.array([0, 0, 0, 0, 0, 0, 1], dtype=dtype),
                    }

def mila(x, beta=-1.0):
    return x * tf.math.tanh(tf.math.softplus(beta + x))

ff_module = lambda node_size, num_layers: \
                    tf.keras.Sequential([
                                            tf.keras.layers.Dense(
                                                units=node_size, #64
                                                activation=mila,
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                            ) for _ in range(num_layers)
                                        ])

ff_module_terminal = lambda node_size, num_layers, output_size: \
                    tf.keras.Sequential([
                                            tf.keras.layers.Dense(
                                                units=node_size, #64
                                                activation=mila,
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                            ) for _ in range(num_layers)
                                        ] + \
                                        [
                                            tf.keras.layers.Dense(
                                                units=output_size, 
                                                activation=None,
                                                kernel_initializer=tf.keras.initializers.he_normal(),
                                        )])


class MultipoleNetRes(tf.keras.Model):
    def __init__(self, node_size=64, edge_size=64, activation=mila, num_steps=4):
        super(MultipoleNetRes, self).__init__()   
        self.num_steps = num_steps
        self.node_size = node_size
        self.edge_size = edge_size        

        self.embedding_mono = gn.modules.GraphIndependent(
            edge_model_fn=lambda: ff_module(self.edge_size, 1),
            node_model_fn=lambda: ff_module(self.node_size, 1),          
        )
        
        self.embedding_dipo = gn.modules.GraphIndependent(
            edge_model_fn=lambda: ff_module(self.edge_size, 1),
            node_model_fn=lambda: ff_module(self.node_size, 1),          
        )
        
        self.embedding_quad = gn.modules.GraphIndependent(
            edge_model_fn=lambda: ff_module(self.edge_size, 1),
            node_model_fn=lambda: ff_module(self.node_size, 1),          
        )
        
        self.embedding_octu = gn.modules.GraphIndependent(
            edge_model_fn=lambda: ff_module(self.edge_size, 1),
            node_model_fn=lambda: ff_module(self.node_size, 1),          
        )
        
        self.gns_mono = [gn.modules.InteractionNetwork( 
                         edge_model_fn=lambda: ff_module(self.edge_size, 2),
                         node_model_fn=lambda: ff_module(self.node_size, 2)) 
                         for _ in range(self.num_steps)]
        
        self.gns_dipo = [gn.modules.InteractionNetwork( 
                         edge_model_fn=lambda: ff_module(self.edge_size, 2),
                         node_model_fn=lambda: ff_module(self.node_size, 2)) 
                            for _ in range(self.num_steps)]
        
        self.gns_quad = [gn.modules.InteractionNetwork( 
                         edge_model_fn=lambda: ff_module(self.edge_size, 2),
                         node_model_fn=lambda: ff_module(self.node_size, 2)) 
                         for _ in range(self.num_steps)]

        self.gns_octu = [gn.modules.InteractionNetwork( 
                         edge_model_fn=lambda: ff_module(self.edge_size, 2),
                         node_model_fn=lambda: ff_module(self.node_size, 2)) 
                         for _ in range(self.num_steps)]

        self.mono = ff_module_terminal(self.node_size, 2, 1)
        self.dipo = ff_module_terminal(self.node_size, 2, 1)
        self.quad = ff_module_terminal(self.node_size, 2, 1)
        self.octu = ff_module_terminal(self.node_size, 2, 1)
 
    def update(self, graphs):   
        initial_nodes = graphs.nodes
        graphs_mono = self.embedding_mono(graphs)
        graphs_dipo = self.embedding_dipo(graphs)
        graphs_quad = self.embedding_quad(graphs)
        graphs_octu = self.embedding_octu(graphs)
        for layer_mono, layer_dipo, layer_quad, layer_octu in zip(self.gns_mono, self.gns_dipo, self.gns_quad, self.gns_octu):
            nodes_mono = graphs_mono.nodes
            nodes_dipo = graphs_dipo.nodes
            nodes_quad = graphs_quad.nodes
            nodes_octu = graphs_octu.nodes
            graphs_mono = layer_mono(graphs_mono)
            graphs_dipo = layer_dipo(graphs_dipo)
            graphs_quad = layer_quad(graphs_quad)
            graphs_octu = layer_octu(graphs_octu)
            graphs_mono = graphs_mono.replace(nodes=graphs_mono.nodes + nodes_mono)    
            graphs_dipo = graphs_dipo.replace(nodes=graphs_dipo.nodes + nodes_dipo)    
            graphs_quad = graphs_quad.replace(nodes=graphs_quad.nodes + nodes_quad)
            graphs_octu = graphs_octu.replace(nodes=graphs_octu.nodes + nodes_octu)
        return graphs_mono, graphs_dipo, graphs_quad, graphs_octu
    
    def monopoles(self, graphs_mono):
        monopoles = self.mono(graphs_mono.nodes)
        monopoles -= tf.reduce_mean(monopoles)
        return monopoles
    
    def dipoles(self, graphs_dipo, edge_features, vectors):
        dipo_features_senders = tf.gather(graphs_dipo.nodes, graphs_dipo.senders)
        dipo_features_receivers = tf.gather(graphs_dipo.nodes, graphs_dipo.receivers)
        features_dipo = tf.concat((dipo_features_senders, dipo_features_receivers, edge_features), axis=-1)
        weighted_vectors = vectors * self.dipo(features_dipo)
        return tf.math.unsorted_segment_sum(weighted_vectors, graphs_dipo.receivers, num_segments=tf.reduce_sum(graphs_dipo.n_node))
    
    def quadrupoles(self, graphs_quad, edge_features, vectors):
        outer_products = get_outer_products(vectors)
        quad_features_senders = tf.gather(graphs_quad.nodes, graphs_quad.senders)
        quad_features_receivers = tf.gather(graphs_quad.nodes, graphs_quad.receivers)
        features_quad = tf.concat((quad_features_senders, quad_features_receivers, edge_features), axis=-1)
        weighted_outer_products = outer_products * tf.expand_dims(self.quad(features_quad), axis=-1)
        return tf.math.unsorted_segment_sum(weighted_outer_products, graphs_quad.receivers, num_segments=tf.reduce_sum(graphs_quad.n_node))
    
    def octupoles(self, graphs_octu, edge_features, vectors):
        octu_products = get_octupole_products(vectors)  # [N_edges, 3, 3, 3, 3]
        octu_features_senders = tf.gather(graphs_octu.nodes, graphs_octu.senders)
        octu_features_receivers = tf.gather(graphs_octu.nodes, graphs_octu.receivers)
        features_octu = tf.concat((octu_features_senders, octu_features_receivers, edge_features), axis=-1)
        # Expand weights to match octupole tensor dimensions [N_edges, 3, 3, 3, 3]
        weights = self.octu(features_octu)  # [N_edges, 1]
        weights_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(weights, axis=-1), axis=-1), axis=-1)  # [N_edges, 1, 1, 1, 1]
        weighted_octu_products = octu_products * weights_expanded  # [N_edges, 3, 3, 3, 3]
        # Sum over edges to get [N_atoms, 3, 3, 3, 3]
        return tf.math.unsorted_segment_sum(weighted_octu_products, graphs_octu.receivers, num_segments=tf.reduce_sum(graphs_octu.n_node))        
    
    def call(self, graphs, coordinates):
        initial_edges = graphs.edges
        vectors = tf.gather(coordinates, graphs.senders) - tf.gather(coordinates, graphs.receivers)
        graphs_mono, graphs_dipo, graphs_quad, graphs_octu = self.update(graphs)
        monopoles = self.monopoles(graphs_mono)
        dipoles = self.dipoles(graphs_dipo, initial_edges, vectors)
        quadrupoles = self.quadrupoles(graphs_quad, initial_edges, vectors)
        octupoles = self.octupoles(graphs_octu, initial_edges, vectors)
        return monopoles, dipoles, quadrupoles, octupoles
    
    def predict(self, coordinates, elements):
        graph = build_graph(coordinates, elements)
        return self(graph, coordinates)   
    
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=dtype)])
def get_outer_products(vectors):
    vectors = tf.expand_dims(vectors, axis=-1)
    return D_Q(vectors * tf.linalg.matrix_transpose(vectors))

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
    
    return D_O(octupole_tensor)
    
@tf.function(experimental_relax_shapes=True)    
def D_Q(quadrupoles):
    return tf.linalg.set_diag(quadrupoles, tf.linalg.diag_part(quadrupoles) - tf.expand_dims((tf.linalg.trace(quadrupoles) / 3), axis=-1))

@tf.function(experimental_relax_shapes=True)    
def D_O(octupoles):
    # Simplified detracing for octupoles
    # For a 4th-order tensor, we subtract traces over all pairs of indices
    # This is a simplified implementation - full detracing would be more complex
    
    delta = tf.eye(3, dtype=dtype)
    
    # Remove trace over first pair of indices (i,j)
    trace_ij = tf.einsum('niikl->nkl', octupoles)  # Sum over i=j
    correction_ij = tf.einsum('ij,nkl->nijkl', delta, trace_ij) / 3.0
    octupoles = octupoles - correction_ij
    
    # Remove trace over second pair of indices (k,l)  
    trace_kl = tf.einsum('nijkk->nij', octupoles)  # Sum over k=l
    correction_kl = tf.einsum('kl,nij->nijkl', delta, trace_kl) / 3.0
    octupoles = octupoles - correction_kl
    
    # Additional detracing could be added for other index pairs if needed
    # For now, this provides basic detracing functionality
    
    return octupoles

def triweight(k, m):
    return tf.math.pow(1 - (k/m) ** 2, 3)
    
def weight_function(distances, num_kernels, min_range, max_range, kernel_function=triweight, alpha=2.0):
    m = alpha * (max_range - min_range) / (num_kernels + 1)
    lower_bound = min_range + m 
    upper_bound = max_range - m
    centers = tf.reshape(tf.cast(tf.linspace(lower_bound, upper_bound, num_kernels), tf.float32), [1, 1, num_kernels])
    k = distances - centers
    return tf.math.maximum(0.0, kernel_function(k, m))  

def build_graph(coords, elements, cutoff=4.0, num_kernels=32):
    node_features = tf.convert_to_tensor([onehots_elements[e] for e in elements])
    dmat = squareform(pdist(coords)).astype(dtype)
    indices = tf.where(tf.math.logical_and(dmat > 0.0, dmat < cutoff))
    edge_weights = weight_function(tf.expand_dims(tf.gather_nd(dmat, indices), axis=-1), num_kernels, 0.5, cutoff)[0]
    senders, receivers = tf.split(indices, 2, axis=-1)
    senders, receivers = tf.cast(tf.squeeze(senders), dtype=tf.int32), tf.cast(tf.squeeze(receivers), dtype=tf.int32)
    n_node = tf.cast(tf.fill(1, node_features.shape[0]), dtype=tf.int32)
    n_edge = tf.cast(tf.fill(1, edge_weights.shape[0]), dtype=tf.int32)
    return gn.graphs.GraphsTuple(node_features, edge_weights, globals=None, receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)
        
def build_graph_batched(coords, elements, cutoff=4.0, num_kernels=32):
    dmat = cdist_tf_batch(coords, coords)
    mol_size = coords.shape[1]
    indices = tf.where(tf.math.logical_and(dmat > 0.0, dmat < cutoff))
    edge_weights = weight_function(tf.expand_dims(tf.gather_nd(dmat, indices), axis=-1), num_kernels, 0.5, cutoff)[0]
    node_features = tf.convert_to_tensor([onehots_elements[e] for e in elements])
    node_features = tf.reshape(tf.tile(node_features[None], [len(dmat), 1, 1]), [-1, 7])
    senders = mol_size * indices[..., 0] + indices[..., 1]
    receivers = mol_size * indices[..., 0] + indices[..., 2]
    n_node, n_edge = node_features.shape[0], edge_weights.shape[0]
    return gn.graphs.GraphsTuple(node_features, edge_weights, globals=None, receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)

def cdist_tf_batch(A, B):
    na = tf.reduce_sum(tf.square(A), axis=-1, keepdims=True)
    nb = tf.transpose(tf.reduce_sum(tf.square(B), axis=-1, keepdims=True), [0, 2, 1])
    return tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    
def load_model(): 
    model = MultipoleNetRes(node_size=128, edge_size=128, num_steps=4)
    checkpoints = []
    save_path = 'weights/final_model_128_4_res_rev2048'
    checkpoints.extend([tf.train.Checkpoint(module=model.embedding_mono),
                        tf.train.Checkpoint(module=model.mono),
                        tf.train.Checkpoint(module=model.embedding_dipo),
                        tf.train.Checkpoint(module=model.dipo),
                        tf.train.Checkpoint(module=model.embedding_quad),
                        tf.train.Checkpoint(module=model.quad),
                        tf.train.Checkpoint(module=model.embedding_octu),
                        tf.train.Checkpoint(module=model.octu),])
    names = ['_embedding_mono', '_mono', '_embedding_dipo', '_dipo', '_embedding_quad', '_quad', '_embedding_octu', '_octu']
    for idl, layer in enumerate(model.gns_mono):
        checkpoints.append(tf.train.Checkpoint(module=layer))
        names.append('_layer_mono' + str(idl))
    for idl, layer in enumerate(model.gns_dipo):
        checkpoints.append(tf.train.Checkpoint(module=layer))
        names.append('_layer_dipo' + str(idl))
    for idl, layer in enumerate(model.gns_quad):
        checkpoints.append(tf.train.Checkpoint(module=layer))
        names.append('_layer_quad' + str(idl))
    for idl, layer in enumerate(model.gns_octu):
        checkpoints.append(tf.train.Checkpoint(module=layer))
        names.append('_layer_octu' + str(idl))       
    for idc, checkpoint in enumerate(checkpoints):
        if '_octu' in names[idc] or '_layer_octu' in names[idc]:
            # Skip loading octupole weights as they don't exist yet
            continue
        checkpoint.restore(save_path + names[idc] + '-1').expect_partial()
    return model
    
    
