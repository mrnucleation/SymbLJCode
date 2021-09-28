import sys
import os
from math import log
from NNetCalc import nParameters
from NNetCalc import runsim as objfunc

from StephPlex import StephSimplex

filename = sys.argv[1]
par = []
with open(filename, 'r') as parfile:
    for line in parfile:
        newpar = float(line.split()[0])
        par.append(newpar)

nParameters
ubounds = [ 4.1 for x in range(nParameters)]
lbounds = [-4.1 for x in range(nParameters)]

startsize = [(ub-lb)*0.05 for lb, ub in zip(lbounds, ubounds)]

simplex = StephSimplex(objfunc)
results, score = simplex.runopt(
                   lbounds, 
                   ubounds, 
                   initialguess=par, 
                   maxeval=100000, 
                   delmin=[1e-13 for x in lbounds],
                   startstepsize=startsize
                   )
print("Post Minimization Score: %s"%(score))
print("Post Minimization Parameters: %s"%(results))
