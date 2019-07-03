import os,sys
import numpy as np
from pysat.solvers import Glucose3


def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        
        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end while

        # Parse diff edge
        while 'DIFF_EDGE' not in line: line = f.readline();
        diff_edge = [ int(x) for x in f.readline().split() ]

        # Parse target cost
        while 'CHROM_NUMBER' not in line: line = f.readline();
        chrom_number = int(f.readline().strip())

    #end
    return Ma,chrom_number,diff_edge

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def parse_glucose(M, ncolors, path, filename):
  g = Glucose3()
  f = open( path+"/"+filename, "w")
  
  Vars = np.arange( 1, M.shape[0]*ncolors+1).reshape(M.shape[0], ncolors)
  #print("nv = {nv}".format(nv=M.shape[0]))
  
  for i in range(M.shape[0]):
    c = list()
    for j in range(ncolors):
      c.append( int(Vars[i,j]) )
    s = ' '.join(map(str,c))
    f.write(s+" 0\n")
    g.add_clause(c)
  
  edges = [ (i1,i2) for i1 in range(M.shape[0]) for i2 in range(i1+1,M.shape[0]) if M[i1,i2] == 1]
  #print("ne = {ne}".format(ne = len(edges)))
  #Ensure each edge has its endpoints colored with different colors
  for _,(i1,i2) in enumerate(edges):
    for j in range(ncolors):
      c = [ - int(Vars[i1,j]), - int(Vars[i2,j]) ]
      s = ' '.join(map(str,c))
      f.write(s+" 0\n")
      g.add_clause( c )
      
  #A given vertex is colored with one color at most 
  for i in range(M.shape[0]):
    for j in range(ncolors-1):
      c = [ - int(Vars[i,j]), - int(Vars[i,j+1]) ]
      s = ' '.join(map(str,c))
      f.write(s+" 0\n")
      g.add_clause( c )
  
  #Symmetry breaking
  i1,i2 = edges[0]
  
  c = [ int(Vars[i1,0]) ]
  g.add_clause( c )
  s = ' '.join(map(str,c))
  f.write(s+" 0\n")
  
  c = [ int(Vars[i2,1]) ]
  g.add_clause( c )
  s = ' '.join(map(str,c))
  f.write(s+" 0\n")
  f.close()
  
  n_clauses = g.nof_clauses()
  n_vars = g.nof_vars()
  
  if "unsat" in path:
    firstline = "p cnf {n_vars} {n_clauses} 0".format(n_vars=n_vars, n_clauses=n_clauses)
  else:
    firstline = "p cnf {n_vars} {n_clauses} 1".format(n_vars=n_vars, n_clauses=n_clauses)

  line_prepender(path+"/"+filename, firstline)


if __name__ == "__main__":

  graph_path = "../adversarial-training"
  sat_path = "../adversarial-training-cnf/sat"
  unsat_path = "../adversarial-training-cnf/unsat"
  
  if not os.path.exists(sat_path):
    os.makedirs(sat_path)
  if not os.path.exists(unsat_path):
    os.makedirs(unsat_path)


  for filename in os.listdir(graph_path):
    if filename.endswith(".graph"):
      Ma, cn, diff_edge = read_graph(graph_path+"/"+filename)
      
      
      parse_glucose(Ma, cn, sat_path, filename.replace(".graph", ".cnf"))
      
     
      Ma_unsat = Ma.copy()
      Ma_unsat[ diff_edge[0], diff_edge[1] ] = Ma_unsat[ diff_edge[1], diff_edge[0] ] = 1
      parse_glucose(Ma_unsat, cn, unsat_path, filename.replace(".graph", ".cnf"))
      
    #end if
  #end for
  
  
#end main
      
