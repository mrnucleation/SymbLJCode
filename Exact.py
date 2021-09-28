import os
from random import random
import sys
import numpy as np

from NNetCalc import runsim
from NNetCalc import activation_funcs, RCUT, maxsize
from FF_Model import ForcefieldModel



dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs)
curweight = dummymodel.get_npweights()
parameters = []


curweight[0][:,:] = 0.0
curweight[0][0,2] = 1.0
curweight[0][0,4] = 1.0
curweight[1][2,0] = 4.0
curweight[1][4,0] = -4.0
curweight[2][0] = 1.0


for i, row in enumerate(curweight):
    for j, x in np.ndenumerate(row):
        print(i, j, x)
        parameters.append(x)

print(parameters)

score = runsim(parameters, verbose=True, usemask=False)

with open("Solution.dat", "w") as outfile:
    for x in parameters:
        outfile.write("%s\n"%(x))

