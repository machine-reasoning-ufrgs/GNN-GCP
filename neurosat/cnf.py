import pycosat
import copy
import numpy as np
import os

class CNF(object):

  def __init__(self,n,m=0):
    self.n = n
    self.m = m
    self.clauses = []
    self.sat = None
  #end

  def SR(n):

    cnf = CNF(n)
    sat = True

    while sat:
      # Select a random k ~ Bernouilli(0.3) + Geo(0.4)
      k = np.random.binomial(1,0.4) + np.random.geometric(0.4)
      # Create a clause with k randomly selected variables
      clause = [ int(np.random.randint(1,n+1) * np.random.choice([-1,+1])) for i in range(k) ]
      # Append clause to cnf
      cnf.clauses.append(clause)
      # Check for satisfiability
      if pycosat.solve(cnf.clauses) == "UNSAT":
        sat = False
        # Create an identical copy of cnf
        cnf2 = copy.deepcopy(cnf)
        # Flip the polarity of a single literal in the last clause of cnf2
        cnf2.clauses[-1][np.random.randint(0,len(cnf2.clauses[-1]))] *= -1
      #end
    #end

    cnf.sat = False
    cnf2.sat = True

    cnf.m = cnf2.m = len(cnf.clauses)

    return cnf,cnf2
  #end

  def SRU(n0,n1):
    n = np.random.randint(n0,n1+1)
    return CNF.SR(n)
  #end

  def random_3SAT_critical(n):
    m = int(4.26 * n)

    cnf = CNF(n,m)

    for i in range(m):
      clause = [ int(np.random.randint(1,n+1) * np.random.choice([-1,+1])) for k in range(3) ]
      cnf.clauses.append( clause )
    #end

    cnf.sat = pycosat.solve(cnf.clauses) != "UNSAT"

    return cnf
  #end

  def write_dimacs(self,path):
    with open(path,"w") as out:
      out.write("p cnf {} {} {}\n".format(self.n, self.m, int(self.sat) ))

      for clause in self.clauses:
        out.write( ' '.join([ str(x) for x in clause]) + ' 0\n')
      #end
    #end
  #end

  def read_dimacs(path):
    with open(path,"r") as f:
      n, m, sat = [ int(x) for x in f.readline().split()[2:]]
      cnf = CNF(n,m)
      cnf.sat = bool(sat)
      for i in range(m):
        cnf.clauses.append( [ int(x) for x in f.readline().split()[:-1]] )
      #end
    #end
    return cnf
  #end
#end

class BatchCNF(object):

  def __init__(self,n,m,clauses,sat):
    """
      batch_size: number of instances in this batch
      n: number of variables for each instance
      total_n: total number of variables among all instances
      m: number of clauses for each instance
      total_m: total number of clauses among all instances
      clauses: concatenated list of clauses among all instances
      sat: satisfiability of each instance
    """
    self.batch_size = len(n)
    self.n = n
    self.total_n = sum(n)
    self.m = m
    self.total_m = sum(m)
    self.clauses = clauses
    self.sat = sat
  #end
  
  def get_dense_matrix(self):
    M = np.zeros( (2*self.total_n, self.total_m), dtype=np.float32 )
    n_cells = sum([ len(clause) for clause in self.clauses ])
    cell = 0
    for (j,clause) in enumerate(self.clauses):
      for literal in clause:
        i = int(abs(literal) - 1)
        p = np.sign(literal)
        if p == +1:
          M[i,j] = 1
        elif p == -1:
          M[self.total_n + i, j] = 1
        #end
        cell += 1
      #end for literal
    #end for j,clause
    return M
  #end get_dense_matrix

  def get_sparse_matrix(self):
    """
      First we need to count the number of non-null cells in our
      adjacency matrix. This can be computed as the sum of all clause
      sizes Σ|c| ∀c ∈ F
    """
    n_cells = sum([ len(clause) for clause in self.clauses ])

    # Define sparse_M with shape (n_cells,2)
    sparse_M = np.zeros((n_cells,2), dtype=np.int)

    cell = 0
    for (j,clause) in enumerate(self.clauses):
      for literal in clause:
        i = int(abs(literal) - 1)
        p = np.sign(literal)

        if p == +1:
          sparse_M[cell] = [i,j]
        elif p == -1:
          sparse_M[cell] = [self.total_n + i, j]
        #end

        cell += 1
      #end
    #end
    return sparse_M, np.ones( n_cells, dtype = np.float32 ), (2*self.total_n, self.total_m)
  #end

#end

def create_batchCNF(instances):
  """
    Create a BatchCNF object from a list of cnf instances
  """
  n = []
  m = []
  clauses = []
  sat = []
  offset = 0
  for cnf in instances:
    n.append(cnf.n)
    m.append(cnf.m)
    clauses.extend( [ [ np.sign(literal) * (abs(literal) + offset) for literal in clause ] for clause in cnf.clauses ] )
    sat.append(cnf.sat)
    offset += cnf.n
  #end

  return BatchCNF(n,m,clauses,sat)
#end

def create_dataset( n_min = 10, n_max = 40, samples = 1000, path = "instances" ):
  for i in range(samples):
    cnf1, cnf2 = CNF.SRU( 10, 40 )
    cnf1.write_dimacs("{}/unsat/{}.cnf".format(path,i))
    cnf2.write_dimacs("{}/sat/{}.cnf".format(path,i))
  #end for
#end

def create_critical_dataset( n = 40, samples = 512, path = "critical_instances" ):
  for i in range( samples ):
    cnf = CNF.random_3SAT_critical( n )
    cnf.write_dimacs( "{}/{}.cnf".format( path, i ) )
  #end for
#end create_critical_dataset

def ensure_datasets( make_critical = False ):
  idirs = [ "instances", "instances/sat", "instances/unsat" ]
  if not all( map( os.path.isdir, idirs ) ):
    for d in idirs:
      os.makedirs( d )
    #end for
    create_dataset( 10, 40, 25600, path = idirs[0] )
  #end if
  tdirs = [ "test-instances", "test-instances/sat", "test-instances/unsat" ]
  if not all( map( os.path.isdir, tdirs ) ):
    for d in tdirs:
      os.makedirs( d )
    #end for
    create_dataset( 40, 40, 512, path = tdirs[0])
  #end if
  if make_critical:
    c40dir = "critical-instances-40"
    if not os.path.isdir( c40dir ):
      os.makedirs( c40dir )
      create_critical_dataset( 40, 512, c40dir )
    #end if
    c80dir = "critical-instances-80"
    if not os.path.isdir( c80dir ):
      os.makedirs( c80dir )
      create_critical_dataset( 80, 512, c80dir )
    #end if
  #end if
#end ensure_datasets

if __name__ == '__main__':
  ensure_datasets()
