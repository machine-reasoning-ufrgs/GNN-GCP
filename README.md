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
```console
foo@bar:~$ whoami
foo
```
