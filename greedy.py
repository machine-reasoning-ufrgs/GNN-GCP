import math
import numpy as np
import random
import networkx as nx


class Greedy(object):
  def __init__(self, Ma):
    self.Ma = Ma
    self.n = Ma.shape[0]
    self.solution = np.zeros( self.n ) - 1  #no color assigned to any vertex
    
  def execute(self):
    self.solution[0] = 0 #first color to first vertex
    
    all_colors = set( range(self.n) )
    unavailable = set()
    for i in range(1, self.n):
      for j in range( 0, self.n):
        if self.Ma[i,j] == 1:
          if self.solution[j] != -1:
            unavailable.add( self.solution[j] )
          #end if
        #end if
      #end for
      available = all_colors - unavailable
      self.solution[i] = np.amin( np.array( list( available )))
      unavailable.clear()
    #end for
    for i in range(self.n):
      for j in range( i + 1, self.n):
        if self.Ma[i,j] == 1 and self.solution[i] == self.solution[j]:
          return -1
    
    
    return np.amax( self.solution) + 1
           
if __name__ == "__main__":
  g = nx.chvatal_graph()
  g = nx.to_numpy_matrix( g )
  greedy = Greedy( g )
  print( greedy.execute() ) 
            
          
      
