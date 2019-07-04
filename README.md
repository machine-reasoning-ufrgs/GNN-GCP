# GNN-GCP
Graph Neural Network architecture to solve the decision version of the graph coloring problem (GCP) (i.e. "is it possible
to colour the given graph with C colours?"). Both training and testing procedures are ran on the verge of satisfiability, that is,
we designed our instances with their exact chromatic number (positive ones) and we find their frozen edges to generate a negative
instance (keeping the previous chromatic number and inserting the frozen edge into the graph).


### Dependencies
```
Python 3.5.2
Tensorflow 1.10.0
Networkx 2.1 (plus Grinpy)
Pycosat 0.6.3
Pysat 0.1.3
Numpy 1.14.5
Ortools 6.10

```
### Running
First, generate the train and test instances (number of samples and other parameters can be changed via argparse options)*
```console
foo@bar:~$ python3 dataset.py -path adversarial-training --train
foo@bar:~$ python3 dataset.py -path adversarial-testing
```
Then, train and test (change epoch to the desired one) the model. 
```console
foo@bar:~$ python3 run_model.py --train --save
foo@bar:~$ python3 run_model.py -loadpath training/checkpoints/epoch\=200.0 --load
```

To train and test Neurosat we first parse the instances to CNF format. 
```console
foo@bar:~$ cd neurosat
foo@bar:~$ python3 parse_to_cnf.py
foo@bar:~$ python3 neurosat_train.py
foo@bar:~$ python3 neurosat_test.py
```

*Obs.: The process of generating train and test instances usually takes very long, so we also made them available [here] (https://drive.google.com/open?id=1wu_pgoaWxUAkselmwoXy8DlrMmbpTavM)

### Publication
The results from this experiment are reported in the research paper ["Graph Colouring Meets Deep Learning: Effective Graph Neural Network Models for Combinatorial Problems"](https://arxiv.org/abs/1903.04598) by [H. Lemos](http://dblp.org/pers/hd/l/Lemos:Henrique), [M. Prates](http://dblp.org/pers/hd/p/Prates:Marcelo_O=_R=), [P. Avelar](http://dblp.org/pers/hd/a/Avelar:Pedro_H=_C=),and [L. Lamb](http://dblp.org/pers/hd/l/Lamb:Lu=iacute=s_C=).
