#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os, time, random, argparse, timeit
import tensorflow as tf
import numpy as np
from itertools import islice
from functools import reduce

from model import build_network
from instance_loader import InstanceLoader
from util import load_weights, save_weights
from tabucol import tabucol, test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_training_batch(sess, model, batch, batch_i, epoch_i, time_steps, d, verbose=True):

    M, C, VC, cn_exists, n_vertices, n_edges, f = batch
    #Generate colors embeddings
    ncolors = np.sum(C)
    #We define the colors embeddings outside, randomly. They are not learnt by the GNN (that can be improved)
    colors_initial_embeddings = np.random.rand(ncolors,d)
    
    # Define feed dict
    feed_dict = {
        model['M']: M,
        model['VC']: VC,
        model['chrom_number']: C,
        model['time_steps']: time_steps,
        model['cn_exists']: cn_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges,
        model['colors_initial_embeddings']: colors_initial_embeddings
    }

    outputs = [model['train_step'], model['loss'], model['acc'], model['predictions'], model['TP'], model['FP'], model['TN'], model['FN']]
    

    # Run model
    loss, acc, predictions, TP, FP, TN, FN = sess.run(outputs, feed_dict = feed_dict)[-7:]

    if verbose:
        # Print stats
        print('{train_or_test} Epoch {epoch_i} Batch {batch_i}\t|\t(n,m,batch size)=({n},{m},{batch_size})\t|\t(Loss,Acc)=({loss:.4f},{acc:.4f})\t|\tAvg. (Sat,Prediction)=({avg_sat:.4f},{avg_pred:.4f})'.format(
            train_or_test = 'Train',
            epoch_i = epoch_i,
            batch_i = batch_i,
            loss = loss,
            acc = acc,
            n = np.sum(n_vertices),
            m = np.sum(n_edges),
            batch_size = n_vertices.shape[0],
            avg_sat = np.mean(cn_exists),
            avg_pred = np.mean(np.round(predictions))
            ),
            flush = True
        )
    #end
    return loss, acc, np.mean(cn_exists), np.mean(predictions), TP, FP, TN, FN
#end


def run_test_batch(sess, model, batch, batch_i, time_steps, logfile, runtabu=True):

    M, n_colors, VC, cn_exists, n_vertices, n_edges, f = batch
    
    # Compute the number of problems
    n_problems = n_vertices.shape[0]

    #open up the batch, which contains 2 instances
    for i in range(n_problems):
      n, m, c = n_vertices[i], n_edges[i], n_colors[i]
      conn = m / n
      n_acc = sum(n_vertices[0:i])
      c_acc = sum(n_colors[0:i])
      
      
      #subset adjacency matrix
      M_t = M[n_acc:n_acc+n, n_acc:n_acc+n]
      c = c if i % 2 == 0 else c + 1
      
      gnnpred = tabupred = 999
      for j in range(2, c + 5):
        n_colors_t = j
        cn_exists_t = 1 if n_colors_t >= c else 0
        VC_t = np.ones( (n,n_colors_t) )
        #Generate colors embeddings
        colors_initial_embeddings = np.random.rand(n_colors_t,d)
        
        feed_dict = {
            model['M']: M_t,
            model['VC']: VC_t,
            model['chrom_number']: np.array([n_colors_t]),
            model['time_steps']: time_steps,
            model['cn_exists']: np.array([cn_exists_t]),
            model['n_vertices']: np.array([n]),
            model['n_edges']: np.array([m]),
            model['colors_initial_embeddings']: colors_initial_embeddings
        }

        outputs = [model['loss'], model['acc'], model['predictions'], model['TP'], model['FP'], model['TN'], model['FN'] ]
        
        # Run model - chromatic number or more
        init_time = timeit.default_timer()
        loss, acc, predictions, TP, FP, TN, FN = sess.run(outputs, feed_dict = feed_dict)[-7:]
        elapsed_gnn_time  = timeit.default_timer() - init_time
        gnnpred = n_colors_t if predictions > 0.5 and n_colors_t < gnnpred else gnnpred
        
        # run tabucol
        if runtabu:
          init_time = timeit.default_timer()
          tabu_solution = tabucol(M_t, n_colors_t, max_iterations=1000)
          elapsed_tabu_time  = timeit.default_timer() - init_time
          tabu_sol = 0 if tabu_solution is None else 1
          tabupred = n_colors_t if tabu_sol == 1 and n_colors_t < tabupred else tabupred
      #end for
      logfile.write('{batch_i} {i} {n} {m} {conn} {tstloss} {tstacc} {cn_exists} {c} {gnnpred} {prediction} {gnntime} {tabupred} {tabutime}\n'.format(
        batch_i = batch_i,
        i = i,
        n= n,
        m = m,
        c = c,
        conn = conn,
        cn_exists = cn_exists_t,
        tstloss = loss,
        tstacc = acc,
        gnnpred = gnnpred, 
        prediction = predictions.item(),
        gnntime = elapsed_gnn_time,
        tabupred = tabupred if runtabu else 0,
        tabutime = elapsed_tabu_time if runtabu else 0
        )
      )
      logfile.flush()
    #end for batch
#end

def summarize_epoch(epoch_i, loss, acc, sat, pred, train=False):
    print('{train_or_test} Epoch {epoch_i} Average\t|\t(Loss,Acc)=({loss:.4f},{acc:.4f})\t|\tAvg. (Sat,Pred)=({avg_sat:.4f},{avg_pred:.4f})'.format(
        train_or_test = 'Train' if train else 'Test',
        epoch_i = epoch_i,
        loss = np.mean(loss),
        acc = np.mean(acc),
        avg_sat = np.mean(sat),
        avg_pred = np.mean(pred)
        ),
        flush = True
    )
#end


if __name__ == '__main__':
    
    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-timesteps', default=32, type=int, help='# Timesteps')
    parser.add_argument('-epochs', default=10000, type=int, help='Training epochs')
    parser.add_argument('-batchsize', default=8, type=int, help='Batch size')
    parser.add_argument('-path', default="adversarial-training", type=str, help='Path to instances')
    parser.add_argument('-loadpath', default=".", type=str, help='Path to checkpoints to be loaded')
    parser.add_argument('-seed', type=int, default=42, help='RNG seed for Python, Numpy and Tensorflow')
    parser.add_argument('--load', const=True, default=False, action='store_const', help='Load model checkpoint?')
    parser.add_argument('--save', const=True, default=False, action='store_const', help='Save model?')
    parser.add_argument('--train', const=True, default=False, action='store_const', help='Train?')
    parser.add_argument('--runtabu', const=True, default=False, action='store_const', help='Run tabucol?')

    # Parse arguments from command line
    args = parser.parse_args()

    # Set RNG seed for Python, Numpy and Tensorflow
    random.seed(vars(args)['seed'])
    np.random.seed(vars(args)['seed'])
    tf.set_random_seed(vars(args)['seed'])
    seed = str(vars(args)['seed'])
    # Setup parameters
    d                       = vars(args)['d']
    time_steps              = vars(args)['timesteps']
    epochs_n                = vars(args)['epochs']
    batch_size              = vars(args)['batchsize']
    path                    = vars(args)['path']
    loadpath                = vars(args)['loadpath']
    load_checkpoints        = vars(args)['load']
    save_checkpoints        = vars(args)['save']
    runtabu                 = vars(args)['runtabu']

    train_params = {
        'batches_per_epoch': 128
    }

    test_params = {
        'batches_per_epoch': 1
    }
    
    # Create train and test loaders
    if vars(args)['train']:
        train_loader = InstanceLoader(path)
    else:
        test_loader  = InstanceLoader(path)

    # Build model
    print('Building model ...', flush=True)
    GNN = build_network(d)

    # Comment the following line to allow GPU use
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        # Initialize global variables
        print('Initializing global variables ... ', flush=True)
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        if load_checkpoints: load_weights(sess,loadpath);
        
        if vars(args)['train']:
          ptrain = 'training_'+seed
          if not os.path.isdir(ptrain):
            os.makedirs(ptrain)
          with open(ptrain+'/log.dat','w') as logfile:
              # Run for a number of epochs
              for epoch_i in np.arange(epochs_n):

                  train_loader.reset()

                  train_stats = { k:np.zeros(train_params['batches_per_epoch']) for k in ['loss','acc','sat','pred','TP','FP','TN','FN'] }
                  

                  print('Training model...', flush=True)
                  for (batch_i, batch) in islice(enumerate(train_loader.get_batches(batch_size)), train_params['batches_per_epoch']):
                      train_stats['loss'][batch_i], train_stats['acc'][batch_i], train_stats['sat'][batch_i], train_stats['pred'][batch_i], train_stats['TP'][batch_i], train_stats['FP'][batch_i], train_stats['TN'][batch_i], train_stats['FN'][batch_i] = run_training_batch(sess, GNN, batch, batch_i, epoch_i, time_steps, d, verbose=True)
                  #end
                  summarize_epoch(epoch_i,train_stats['loss'],train_stats['acc'],train_stats['sat'],train_stats['pred'],train=True)

                  # Save weights
                  savepath = ptrain+'/checkpoints/epoch={epoch}'.format(epoch=round(200*np.ceil((epoch_i+1)/200)))
                  os.makedirs(savepath, exist_ok=True)
                  if save_checkpoints: save_weights(sess, savepath);

                  logfile.write('{epoch_i} {trloss} {tracc} {trsat} {trpred} {trTP} {trFP} {trTN} {trFN} \n'.format(
                      
                      epoch_i = epoch_i,

                      trloss = np.mean(train_stats['loss']),
                      tracc = np.mean(train_stats['acc']),
                      trsat = np.mean(train_stats['sat']),
                      trpred = np.mean(train_stats['pred']),
                      trTP = np.mean(train_stats['TP']),
                      trFP = np.mean(train_stats['FP']),
                      trTN = np.mean(train_stats['TN']),
                      trFN = np.mean(train_stats['FN']),

                      )
                  )
                  logfile.flush()
              #end
          #end
        else:
          if not os.path.isdir('testing_'+seed):
            os.makedirs('testing_'+seed)
          with open('testing_'+seed+'/log.dat','w') as logfile:
             
            test_loader.reset()
            logfile.write('batch instance vertices edges connectivity loss acc sat chrom_number gnnpred gnncertainty gnntime tabupred tabutime\n')
            print('Testing model...', flush=True)
            for (batch_i, batch) in enumerate(test_loader.get_test_batches(1,2048)):
                run_test_batch(sess, GNN, batch, batch_i, time_steps, logfile,runtabu)
            #end
            logfile.flush()
                  
    #end
#end



