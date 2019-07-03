import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
# Import model builder
from model import build_neurosat
# Import tools
from cnf import ensure_datasets
import instance_loader
import itertools
from util import timestamp, memory_usage
from logutil import run_and_log_batch

if __name__ == '__main__':
  if not os.path.isdir( "tmp-64" ):
    os.makedirs( "tmp-64" )
  #end if
  epochs = 10000
  d = 64 
  
  time_steps = 32
  batch_size = 4
  batches_per_epoch = 128
  
  early_stopping_window = [ 0 for _ in range(3) ]
  early_stopping_threshold = 0.85

  # Build model
  print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
  solver = build_neurosat( d )

  # Create batch loader
  print( "{timestamp}\t{memory}\tLoading instances ...".format( timestamp = timestamp(), memory = memory_usage() ) )
  generator = instance_loader.InstanceLoader( "../adversarial-training-cnf" )
  
  # Create model saver
  saver = tf.train.Saver()

  # Disallow GPU use
  config = tf.ConfigProto( device_count = {"GPU":0})
  with tf.Session(config=config) as sess:

    # Initialize global variables
    print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    sess.run( tf.global_variables_initializer() )
    
    if os.path.exists( "./tmp-64/neurosat.ckpt" ):
      # Restore saved weights
      print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
      saver.restore(sess, "./tmp-64/neurosat.ckpt")
    #end if

    # Run for a number of epochs
    print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
    for epoch in range( epochs ):

      # Save current weights
      save_path = saver.save(sess, "./tmp-64/neurosat.ckpt")
      print( "{timestamp}\t{memory}\tMODEL SAVED IN PATH: {save_path}".format( timestamp = timestamp(), memory = memory_usage(), save_path=save_path ) )
      
      if all( [ early_stopping_threshold < v for v in early_stopping_window ] ):
        print( "{timestamp}\t{memory}\tEARLY STOPPING because the test accuracy on the last {epochs} epochs were above {threshold:.2f}% accuracy.".format( timestamp = timestamp(), memory = memory_usage(), epochs = len( early_stopping_window ), threshold = early_stopping_threshold * 100 ) )
        break
      #end if

      # Reset training generator and run with a sample of the training instances
      print( "{timestamp}\t{memory}\tTRAINING SET BEGIN".format( timestamp = timestamp(), memory = memory_usage() ) )
      generator.reset()
      epoch_loss = 0.0
      epoch_accuracy = 0.0
      for b, batch in itertools.islice( enumerate( generator.get_batches( batch_size ) ), batches_per_epoch ):
        l, a, p = run_and_log_batch( sess, solver, epoch, b, batch, time_steps )
        epoch_loss += l
        epoch_accuracy += a
      #end for
      epoch_loss /= batches_per_epoch
      epoch_accuracy /= batches_per_epoch
    
    #end for
  #end Session
