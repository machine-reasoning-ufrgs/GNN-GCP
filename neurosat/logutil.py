import sys, os, time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import instance_loader
import numpy as np
from util import timestamp, memory_usage

def sigmoid( x, derivative = False ):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
#end sigmoid

def run_and_log_batch( sess, solver, epoch, b, batch, time_steps, train = True ):
  sat = list( 1 if sat else 0 for sat in batch.sat )
  # Build feed_dict
  feed_dict = {
    solver["gnn"].time_steps: time_steps,
    solver["gnn"].matrix_placeholders["M"]: batch.get_dense_matrix(),
    solver["instance_SAT"]: np.array( sat ),
    solver["num_vars_on_instance"]: batch.n
  }
  # Run session
  if train:
    _, pred_SAT, loss_val, accuracy_val = sess.run(
        [ solver["train_step"], solver["predicted_SAT"], solver["loss"], solver["accuracy"] ],
        feed_dict = feed_dict
    )
  else:
    pred_SAT, loss_val, accuracy_val = sess.run(
        [ solver["predicted_SAT"], solver["loss"], solver["accuracy"] ],
        feed_dict = feed_dict
    )
  #end if
  avg_pred = np.mean( np.round( sigmoid( pred_SAT ) ) )
  # Print train step loss and accuracy, as well as predicted sat values compared with the normal ones
  print(
    "{epoch} {batch} {loss:.4f} {accuracy:.4f} {avg_pred:.4f}".format(
      epoch = epoch,
      batch = b,
      loss = loss_val,
      accuracy = accuracy_val,
      avg_pred = avg_pred,
    ),
    flush = True
  )
  return loss_val, accuracy_val, avg_pred
#end run_and_log_batch

def test_with( sess, solver, path, name, time_steps = 26, batch_size = 1 ):
  # Load test instances
  print( "{timestamp}\t{memory}\tLoading test {name} instances ...".format( timestamp = timestamp(), memory = memory_usage(), name = name ) )
  test_generator = instance_loader.InstanceLoader( path )
  test_loss = 0.0
  test_accuracy = 0.0
  test_avg_pred = 0.0
  test_batches = 0
  # Run with the test instances
  print( "{timestamp}\t{memory}\t{name} TEST SET BEGIN".format( timestamp = timestamp(), memory = memory_usage(), name = name ) )
  for b, batch in enumerate( test_generator.get_batches( batch_size ) ):
    l, a, p = run_and_log_batch( sess, solver, name, b, batch, time_steps, train = False )
    test_loss += l
    test_accuracy += a
    test_avg_pred += p
    test_batches += 1
  #end for
  # Summarize results and print test summary
  test_loss /= test_batches
  test_accuracy /= test_batches
  test_avg_pred /= test_batches
  print( "{timestamp}\t{memory}\t{name} TEST SET END Mean loss: {loss:.4f} Mean Accuracy = {accuracy} Mean prediction {avg_pred:.4f}".format(
    loss = test_loss,
    accuracy = test_accuracy,
    avg_pred = test_avg_pred,
    timestamp = timestamp(),
    memory = memory_usage(),
    name = name
    )
  )
#end test_with
