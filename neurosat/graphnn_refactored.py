import tensorflow as tf
from mlp import Mlp

class GraphNN(object):
  def __init__(
    self,
    var,
    mat,
    msg,
    loop,
    MLP_depth = 3,
    MLP_weight_initializer = tf.contrib.layers.xavier_initializer,
    MLP_bias_initializer = tf.zeros_initializer,
    Cell_activation = tf.nn.relu,
    Msg_activation = tf.nn.relu,
    Msg_last_activation = None,
    float_dtype = tf.float32,
    name = 'GraphNN'
  ):
    self.var, self.mat, self.msg, self.loop, self.name = var, mat, msg, loop, name

    self.MLP_depth = MLP_depth
    self.MLP_weight_initializer = MLP_weight_initializer
    self.MLP_bias_initializer = MLP_bias_initializer
    self.Cell_activation = Cell_activation
    self.Msg_activation = Msg_activation
    self.Msg_last_activation  = Msg_last_activation 
    self.float_dtype = float_dtype

    # Check model for inconsistencies
    self.check_model()

    # Build the network
    with tf.variable_scope(self.name):
      with tf.variable_scope('placeholders'):
        self._init_placeholders()
      #end
      with tf.variable_scope('parameters'):
        self._init_parameters()
      #end
      with tf.variable_scope('utilities'):
        self._init_utilities()
      #end
      with tf.variable_scope('run'):
        self._run()
      #end
    #end
  #end

  def check_model(self):
    # Procedure to check model for inconsistencies
    for v in self.var:
      if v not in self.loop:
        raise Warning('Variable {v} is not updated anywhere! Consider removing it from the model'.format(v=v))
      #end
    #end

    for v in self.loop:
      if v not in self.var:
        raise Exception('Updating variable {v}, which has not been declared!'.format(v=v))
      #end
    #end

    for mat, (v1,v2) in self.mat.items():
      if v1 not in self.var:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v1))
      #end
      if v2 not in self.var and type(v2) is not int:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v2))
      #end
    #end

    for msg, (v1,v2) in self.msg.items():
      if v1 not in self.var:
        raise Exception('Message {msg} maps from undeclared variable {v}'.format(msg=msg, v=v1))
      #end
      if v2 not in self.var:
        raise Exception('Message {msg} maps to undeclared variable {v}'.format(msg=msg, v=v2))
      #end
    #end
  #end

  def _init_placeholders(self):
    self.matrix_placeholders = {}
    for m in self.mat:
      if type(self.mat[m][1]) == int:
        self.matrix_placeholders[m] = tf.placeholder(self.float_dtype, shape=(None,self.mat[m][1]), name=m)
      else:
        self.matrix_placeholders[m] = tf.placeholder(self.float_dtype, shape=(None,None), name=m)
      #end
    #end
    self.time_steps = tf.placeholder(tf.int32, shape=(), name='time_steps')
  #end

  def _init_parameters(self):
    # Init embeddings
    self._initial_embeddings = { v:tf.get_variable(initializer=tf.random_normal((1,d)), dtype=self.float_dtype, name='{v}_init'.format(v=v)) for (v,d) in self.var.items() }
    # Init LSTM cells
    self._RNN_cells = { v:tf.contrib.rnn.LayerNormBasicLSTMCell(d, activation=self.Cell_activation) for (v,d) in self.var.items() }
    # Init message-computing MLPs
    self._msg_MLPs = { msg:Mlp(layer_sizes=[self.var[vin]]*self.MLP_depth+[self.var[vout]], activations=[self.Msg_activation]*self.MLP_depth + [self.Msg_last_activation], kernel_initializer=self.MLP_weight_initializer(), bias_initializer=self.MLP_weight_initializer(), name=msg, name_internal_layers=True) for msg, (vin,vout) in self.msg.items() }
  #end

  def _init_utilities(self):
    self.num_vars = {}
    for m, (v1,v2) in self.mat.items():
      if v1 not in self.num_vars:
        self.num_vars[v1] = tf.shape(self.matrix_placeholders[m])[0]
      #end
      if v2 not in self.num_vars:
        self.num_vars[v2] = tf.shape(self.matrix_placeholders[m])[1]
      #end
    #end
  #end

  def _run(self):
    states = {}
    for v, init in self._initial_embeddings.items():
      denom = tf.sqrt(tf.cast(self.var[v], self.float_dtype))
      h0 = tf.tile(tf.div(init,denom), (self.num_vars[v],1))
      c0 = tf.zeros_like(h0, dtype=self.float_dtype)
      states[v] = tf.contrib.rnn.LSTMStateTuple(h=h0, c=c0)
    #end

    def while_body(t,states):
      new_states = {}
      for v in self.var:
        inputs = []
        for update in self.loop[v]:
          if 'var' in update:
            y = states[update['var']].h
            if 'fun' in update: y = update['fun'](y);
            if 'msg' in update: y = self._msg_MLPs[update['msg']](y);
            if 'mat' in update: y = tf.matmul(self.matrix_placeholders[update['mat']], y, adjoint_a='transpose?' in update and update['transpose?'])
            inputs.append(y)
          else:
            inputs.append(self.matrix_placeholders[update['mat']])
          #end
        #end
        inputs = tf.concat(inputs,axis=1)
        with tf.variable_scope('{v}_cell'.format(v=v)):
          _, new_states[v] = self._RNN_cells[v](inputs=inputs, state=states[v])
        #end
      #end
      return (t+1), new_states
    #end
    _, self.last_states = tf.while_loop(
      lambda t, state: tf.less(t,self.time_steps),
      while_body,
      [0,states]
      )
  #end

  def __call__(self, *args, **kwargs):

    return self.last_states
  #end
#end
