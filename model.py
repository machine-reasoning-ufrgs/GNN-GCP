import sys, os
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graphnn import GraphNN
from mlp import Mlp

def build_network(d):

    # Define hyperparameters
    d = d
    learning_rate = 2e-5
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.65

    # Placeholder for answers to the decision problems (one per problem)
    cn_exists = tf.placeholder( tf.float32, shape = (None,), name = 'cn_exists' )
    # Placeholders for the list of number of vertices and edges per instance
    n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')
    n_edges     = tf.placeholder( tf.int32, shape = (None,), name = 'n_edges')
    # Placeholder for the adjacency matrix connecting each vertex to its neighbors 
    M_matrix   = tf.placeholder( tf.float32, shape = (None,None), name = "M" )
    # Placeholder for the adjacency matrix connecting each vertex to its candidate colors
    VC_matrix = tf.placeholder( tf.float32, shape = (None,None), name = "VC" )
    # Placeholder for chromatic number (one per problem)
    chrom_number = tf.placeholder( tf.float32, shape = (None,), name = "chrom_number" )
    # Placeholder for the number of timesteps the GNN is to run for
    time_steps  = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
    #Placeholder for initial color embeddings for the given batch
    colors_initial_embeddings = tf.placeholder( tf.float32, shape=(None,d), name= "colors_initial_embeddings")
    
    
    # All vertex embeddings are initialized with the same value, which is a trained parameter learned by the network
    total_n = tf.shape(M_matrix)[1]
    v_init = tf.get_variable(initializer=tf.random_normal((1,d)), dtype=tf.float32, name='V_init')
    vertex_initial_embeddings = tf.tile(
        tf.div(v_init, tf.sqrt(tf.cast(d, tf.float32))),
        [total_n, 1]
    )
    
    
    
    # Define GNN dictionary
    GNN = {}

    # Define Graph neural network
    gnn = GraphNN(
        {
            # V is the set of vertex embeddings
            'V': d,
            # C is for color embeddings
            'C': d
        },
        {
            # M is a V×V adjacency matrix connecting each vertex to its neighbors
            'M': ('V','V'),
            # MC is a VxC adjacency matrix connecting each vertex to its candidate colors
            'VC': ('V','C')
        },
        {
            # V_msg_C is a MLP which computes messages from vertex embeddings to color embeddings
            'V_msg_C': ('V','C'),
            # C_msg_V is a MLP which computes messages from color embeddings to vertex embeddings
            'C_msg_V': ('C','V')
        },
        {   # V(t+1) <- Vu( M x V, VC x CmsgV(C) )
            'V': [
                {
                    'mat': 'M',
                    'var': 'V'
                },
                {
                    'mat': 'VC',
                    'var': 'C',
                    'msg': 'C_msg_V'
                }
            ],
            # C(t+1) <- Cu( VC^T x VmsgC(V))
            'C': [
                {
                    'mat': 'VC',
                    'msg': 'V_msg_C',
                    'transpose?': True,
                    'var': 'V'
                }
            ]
        }
        ,
        name='graph-coloring'
    )

    # Populate GNN dictionary
    GNN['gnn']          = gnn
    GNN['cn_exists'] = cn_exists
    GNN['n_vertices']   = n_vertices
    GNN['n_edges']      = n_edges
    GNN["M"]           = M_matrix
    GNN["VC"] = VC_matrix
    GNN["chrom_number"] = chrom_number
    GNN["time_steps"]   = time_steps
    GNN["colors_initial_embeddings"] = colors_initial_embeddings

    # Define V_vote, which will compute one logit for each vertex
    V_vote_MLP = Mlp(
        layer_sizes = [ d for _ in range(3) ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = 1,
        name = 'V_vote',
        name_internal_layers = True,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
        )
    
    # Get the last embeddings
    last_states = gnn(
      { "M": M_matrix, "VC": VC_matrix, 'chrom_number': chrom_number },
      { "V": vertex_initial_embeddings, "C": colors_initial_embeddings },
      time_steps = time_steps
    )
    GNN["last_states"] = last_states
    V_n = last_states['V'].h
    C_n = last_states['C'].h
    
    
    # Compute a vote for each embedding
    V_vote = tf.reshape(V_vote_MLP(V_n), [-1])

    # Compute the number of problems in the batch
    num_problems = tf.shape(n_vertices)[0]

    # Compute a logit probability for each problem
    pred_logits = tf.while_loop(
        lambda i, pred_logits: tf.less(i, num_problems),
        lambda i, pred_logits:
            (
                (i+1),
                pred_logits.write(
                    i,
                    tf.reduce_mean(V_vote[tf.reduce_sum(n_vertices[0:i]):tf.reduce_sum(n_vertices[0:i])+n_vertices[i]])
                )
            ),
        [0, tf.TensorArray(size=num_problems, dtype=tf.float32)]
        )[1].stack()
    # Convert logits into probabilities
    GNN['predictions'] = tf.sigmoid(pred_logits)

    # Compute True Positives, False Positives, True Negatives, False Negatives, accuracy
    GNN['TP'] = tf.reduce_sum(tf.multiply(cn_exists, tf.cast(tf.equal(cn_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['FP'] = tf.reduce_sum(tf.multiply(cn_exists, tf.cast(tf.not_equal(cn_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['TN'] = tf.reduce_sum(tf.multiply(tf.ones_like(cn_exists)-cn_exists, tf.cast(tf.equal(cn_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['FN'] = tf.reduce_sum(tf.multiply(tf.ones_like(cn_exists)-cn_exists, tf.cast(tf.not_equal(cn_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['acc'] = tf.reduce_mean(tf.cast(tf.equal(cn_exists, tf.round(GNN['predictions'])), tf.float32))

    # Define loss
    GNN['loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=cn_exists, logits=pred_logits))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(name='Adam', learning_rate=learning_rate)

    # Compute cost relative to L2 normalization
    vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])

    # Define gradients and train step
    grads, _ = tf.clip_by_global_norm(tf.gradients(GNN['loss'] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
    GNN['train_step'] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    
    GNN['C_n'] = C_n
    
    # Return GNN dictionary
    return GNN
#end
