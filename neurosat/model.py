import tensorflow as tf
import numpy as np
# Import model builder
from graphnn_refactored import GraphNN
from mlp import Mlp
from cnf import CNF

def build_neurosat(d):

  # Hyperparameters
  learning_rate = 2e-5
  parameter_l2norm_scaling = 1e-10
  global_norm_gradient_clipping_ratio = 0.65

  # Define neurosat dictionary
  neurosat = {}

  # Define placeholder for satisfiability statuses (one per problem)
  instance_SAT = tf.placeholder( tf.float32, [ None ], name = "instance_SAT" )

  # Define INV, a tf function to exchange positive and negative literal embeddings
  def INV(Lh):
    l = tf.shape(Lh)[0]
    n = tf.div(l,tf.constant(2))
    # Send messages from negated literals to positive ones, and vice-versa
    Lh_pos = tf.gather( Lh, tf.range( tf.constant( 0 ), n ) )
    Lh_neg = tf.gather( Lh, tf.range( n, l ) )
    Lh_inverted = tf.concat( [ Lh_neg, Lh_pos ], axis = 0 )
    return Lh_inverted
  #end

  # Define Graph neural network
  gnn = GraphNN(
    {
      "L": d,
      "C": d
    },
    {
      "M": ("L","C")
    },
    {
      "Lmsg": ("L","C"),
      "Cmsg": ("C","L")
    },
    {
      "L": [
        {
          "fun": INV,
          "var": "L"
        },
        {
          "mat": "M",
          "msg": "Cmsg",
          "var": "C"
        }
      ],
      "C": [
        {
          "mat": "M",
          "transpose?": True,
          "msg": "Lmsg",
          "var": "L"
        }
      ]
    },
    name="NeuroSAT"
    )

  # Define L_vote
  L_vote_MLP = Mlp(
    layer_sizes = [ d for _ in range(3) ],
    activations = [ tf.nn.relu for _ in range(3) ],
    output_size = 1,
    name = "L_vote",
    name_internal_layers = True,
    kernel_initializer = tf.contrib.layers.xavier_initializer(),
    bias_initializer = tf.zeros_initializer()
    )

  # Compute the number of variables
  n = tf.floordiv( tf.shape( gnn.matrix_placeholders["M"] )[0], tf.constant( 2 ) )
  # Compute number of problems
  p = tf.shape( instance_SAT )[0]
  # Compute number of variables per instance
  num_vars_on_instance = tf.placeholder( tf.int32, [ None ], name = "instance_n" )

  # Get the last embeddings
  L_n = gnn.last_states["L"].h
  L_vote = L_vote_MLP( L_n )

  # Reorganize votes' result to obtain a prediction for each problem instance

  def _vote_while_cond(i, p, n_acc, n, n_var_list, predicted_sat, L_vote):
    return tf.less( i, p )
  #end _vote_while_cond

  def _vote_while_body(i, p, n_acc, n, n_var_list, predicted_SAT, L_vote):
    # Helper for the amount of variables in this problem
    i_n = n_var_list[i]
    # Gather the positive and negative literals for that problem
    pos_lits = tf.gather( L_vote, tf.range( n_acc, tf.add( n_acc, i_n ) ) )
    neg_lits = tf.gather( L_vote, tf.range( tf.add( n, n_acc ), tf.add( n, tf.add( n_acc, i_n ) ) ) )
    # Concatenate positive and negative literals and average their vote values
    problem_predicted_SAT = tf.reduce_mean( tf.concat( [pos_lits, neg_lits], axis = 1 ) )
    # Update TensorArray
    predicted_SAT = predicted_SAT.write( i, problem_predicted_SAT )
    return tf.add( i, tf.constant( 1 ) ), p, tf.add( n_acc, i_n ), n, n_var_list, predicted_SAT, L_vote
  #end _vote_while_body
      
  predicted_SAT = tf.TensorArray( size = p, dtype = tf.float32 )
  _, _, _, _, _, predicted_SAT, _ = tf.while_loop(
    _vote_while_cond,
    _vote_while_body,
    [ tf.constant( 0, dtype = tf.int32 ), p, tf.constant( 0, dtype = tf.int32 ), n, num_vars_on_instance, predicted_SAT, L_vote ]
  )
  predicted_SAT = predicted_SAT.stack()

  # Define loss, accuracy
  predict_costs = tf.nn.sigmoid_cross_entropy_with_logits( labels = instance_SAT, logits = predicted_SAT )
  predict_cost = tf.reduce_mean( predict_costs )
  vars_cost = tf.zeros([])
  tvars = tf.trainable_variables()
  for var in tvars:
    vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
  #end for
  loss = tf.add( predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) )
  optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
  grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
  train_step = optimizer.apply_gradients( zip( grads, tvars ) )
  
  accuracy = tf.reduce_mean(
    tf.cast(
      tf.equal(
        tf.cast( instance_SAT, tf.bool ),
        tf.cast( tf.round( tf.nn.sigmoid( predicted_SAT ) ), tf.bool )
      )
      , tf.float32
    )
  )

  neurosat["gnn"] = gnn
  neurosat["instance_SAT"] = instance_SAT
  neurosat["predicted_SAT"] = predicted_SAT
  neurosat["num_vars_on_instance"] = num_vars_on_instance
  neurosat["loss"] = loss
  neurosat["accuracy"] = accuracy
  neurosat["train_step"] = train_step

  return neurosat
#end build_neurosat
