import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
# Import model builder
from model import build_neurosat
# Import tools
from cnf import CNF, BatchCNF, create_batchCNF
import itertools
import numpy as np
from util import timestamp, memory_usage
from logutil import run_and_log_batch, sigmoid
from parse_to_cnf import parse_glucose

def load_weights(sess,path,scope=None):
  if os.path.exists(path):
    # Restore saved weights
    print("Restoring saved model ... ")
    # Create model saver
    if scope is None:
      saver = tf.train.Saver()
    else:
      saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
    #end
    saver.restore(sess, "%s/neurosat.ckpt" % path)
  else:
    raise Exception('Path does not exist!')
  #end if
#end

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        Mw = np.zeros((n,n),dtype=float)

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end

        # Parse diff edge
        while 'DIFF_EDGE' not in line: line = f.readline();
        diff_edge = [ int(x) for x in f.readline().split() ]

        # Parse target cost
        while 'CHROM_NUMBER' not in line: line = f.readline();
        chrom_number = int(f.readline().strip())

    #end
    return Ma,Mw,chrom_number,diff_edge
#end


def run_test_batch( sess, solver, batch, time_steps ):
  sat = list( 1 if sat else 0 for sat in batch.sat )
  # Build feed_dict
  feed_dict = {
    solver["gnn"].time_steps: time_steps,
    solver["gnn"].matrix_placeholders["M"]: batch.get_dense_matrix(),
    solver["instance_SAT"]: np.array( sat ),
    solver["num_vars_on_instance"]: batch.n
  }
  # Run session
  pred_SAT, loss_val, accuracy_val = sess.run(
        [ solver["predicted_SAT"], solver["loss"], solver["accuracy"] ],
        feed_dict = feed_dict
  )
  
  avg_pred = np.mean( np.round( sigmoid( pred_SAT ) ) )
  # Print train step loss and accuracy, as well as predicted sat values compared with the normal ones
  
  return loss_val, accuracy_val, avg_pred
#end run_and_log_batch

if __name__ == '__main__':
  d = 64
  test_folder = "../adversarial-testing"
  
  time_steps = 32
  
  solver = build_neurosat( d )
  # Create model saver
  saver = tf.train.Saver()


  # Disallow GPU use
  config = tf.ConfigProto( device_count = {"GPU":0})
  with tf.Session(config=config) as sess:

    # Initialize global variables
    print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    sess.run( tf.global_variables_initializer() )
    
    if os.path.exists( "./tmp-64/" ):
      # Restore saved weights
      print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
      saver.restore(sess, "./tmp-64/neurosat.ckpt")
    if not os.path.exists("test-tmp"):
      os.makedirs("test-tmp")

    with open("test-64.log", "w") as f:
      f.write("file epoch batchnumber nvertices nedges loss sat cn neurosat_cnpred pred\n")
      for e, filename in enumerate(os.listdir(test_folder)):
        if filename.endswith(".graph"):
          Ma, _, cn, diff_edge = read_graph(test_folder+"/"+filename)
          #first iterate without the diff edge and with cn
          for j in range(2, cn + 5):
            nv = Ma.shape[0]
            ne = len(np.nonzero(Ma)[0])
            parse_glucose(Ma,j, "test-tmp", "temp.cnf")
            batch = create_batchCNF( [CNF.read_dimacs("test-tmp/temp.cnf")] )
            l, a, p = run_test_batch( sess, solver, batch, time_steps )
            if p == 1:
              f.write(
                "{filename} {epoch} {batch} {nv} {ne} {loss:.4f} {accuracy:.4f} {cn} {cnpred} {avg_pred:.4f}\n".format(
                filename = filename,
                epoch = e,
                batch = 0,
                nv = nv,
                ne = ne,
                loss = l,
                accuracy = a,
                cn = cn,
                cnpred = j,
                avg_pred = p,
              ),
              )
              break
            #end if
          #end for
          #now iterate on the second instance with diff_edge and cn = cn+1
          Ma[ diff_edge[0], diff_edge[1] ] = Ma[ diff_edge[1], diff_edge[0] ] = 1
          ne = len(np.nonzero(Ma)[0])
          cn += 1
          for j in range(2, cn + 5):
            parse_glucose(Ma,j, "test-tmp", "temp.cnf")
            batch = create_batchCNF( [CNF.read_dimacs("test-tmp/temp.cnf")] )
            l, a, p = run_test_batch( sess, solver, batch, time_steps )
            if p == 1:
              f.write(
                "{filename} {epoch} {batch} {nv} {ne} {loss:.4f} {accuracy:.4f} {cn} {cnpred} {avg_pred:.4f}\n".format(
                filename = filename,
                epoch = e,
                batch = 1,
                nv = nv,
                ne = ne,
                loss = l,
                accuracy = a,
                cn = cn,
                cnpred = j,
                avg_pred = p,
              ),
              )
              f.flush()
              break
            #end if
          #end for
        #end if
      #end for
    #end with

