import sys
import os
from MCTSOpt import Tree
#from MCTSOpt.ParameterOpt.LogisticSearch import LogisticSearch
#from ParameterObject import ParameterData as LogisticSearch
from LogisticSearch import LogisticSearch
from SelectionRule_UBUnique import UBUnique as UBEnergy 
from math import sqrt

import numpy as np
import tensorflow as tf
from time import time
from datetime import datetime
from NNetCalc import nParameters, zonewidth
from NNetCalc import runsim as objfunc
from NNetCalc import inv_deadzone, xlow, xhigh

from random import random, seed
from copy import deepcopy
seed(datetime.now())

from init_latin_hypercube_sampling import init_latin_hypercube_sampling
def expandhead_latin(npoints, tree, lb, ub):
    seedpoints = init_latin_hypercube_sampling(np.array(lb), np.array(ub), npoints)
    for point in seedpoints:
#        print("Point %s"%(point))
        nodedata = indata.newdataobject()
        nodedata.setstructure(list(point))
        tree.expandfromdata(newdata=nodedata)


def expandhead_radiallatin(npoints, ndim, tree, rmax):
    seedpoints = init_latin_hypercube_sampling(np.array([0.0]), np.array([rmax]), npoints)
    for rcur in seedpoints:
        u = np.random.normal(0.0, 1.0, ndim)  # an array of d normally distributed random variables
        norm = np.sum(u**2) **(0.5)
        x = rcur*u/norm
        nodedata = indata.newdataobject()
        nodedata.setstructure(x)
        tree.expandfromdata(newdata=nodedata)


restartfile = 'tree_nature.restart'
restartrun = os.path.exists(restartfile)
#restartrun = False
depthlimit = 9
try:
    filename = sys.argv[1]
    parameters = []
    with open(filename, 'r') as parfile:
        for line in parfile:
            newpar = float(line.split()[0])
            parameters.append(newpar)
    parameters = [ inv_deadzone(x, xlow, xhigh)  for x in parameters]
    ubounds = [ x+5E-01 for x in parameters]
    lbounds = [ x-5E-01 for x in parameters]
    ubounds = [ 0.0 if x < 0.0 else x for x in parameters]
    lbounds = [ 0.0 if x > 0.0 else x for x in parameters]
    startset = [ x  for x in parameters]
    print("Loaded from Parameters File")
except:
    nParameters
    ubounds = [ 6.0+0.5*zonewidth for x in range(nParameters)]
    lbounds = [-6.0-0.5*zonewidth for x in range(nParameters)]
    startset = [ 0.0  for lb, ub in zip(lbounds, ubounds) ]
depthscale = [10.0, 0.8, 0.5, 0.25, 0.1]
#depthscale = [sqrt(nParameters)*x for x in depthscale]
#[300, 50, 2500, 50, 400, 8, 1]


ubounds = np.array(ubounds)
lbounds = np.array(lbounds)

options ={'verbose':2}

indata = LogisticSearch(parameters=startset, ubounds=ubounds, lbounds=lbounds, lossfunction=objfunc, depthscale=depthscale, options=options)
#objfunc = runsim
#indata.setevaluator(objfunc)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#---Tree Main Run loop---
#Critical Parameters to Set
tree = Tree(seeddata=indata, 
        playouts=5, 
        selectfunction=UBEnergy, 
        headexpansion=2,
        verbose=True)
tree.setconstant(0e0)
starttime = time()
if restartrun:
    tree.loadtree(restartfile, seeddata=indata)
    tree.autoscaleconstant(scaleboost=0.5)
else:
#    expandhead_latin(52, tree, lbounds, ubounds)
    expandhead_radiallatin(22, len(ubounds), tree, 1e-2)
    tree.expandfromdata(newdata=indata)
    tree.expand(nExpansions=1, savefile=restartfile)
#    tree.savetree(restartfile)
lastmin = 1e300
lastloop = 0
tree.setplayouts(40)
tree.setconstant(1.0)

#tree.selectpath()

#tree.minexpand(scaleconst=1e0)
#for iRep in range(30):
for iLoop in range(1,1550):
    print("Loop Number: %s"%(iLoop))
#    curtime = time()
#    print("Search Duration: %s"%(curtime-starttime))
    print("Loop Number: %s"%(iLoop))
#    print("Search Duration: %s"%(curtime-starttime))
    for i in range(5):
        tree.playexpand(nExpansions=2, depthlimit=depthlimit, savefile=restartfile, allonenode=True)
        tree.simulate(nSimulations=2)
        if i%4 != 3:
            tree.autoscaleconstant(scaleboost=2.5)
        else:
            tree.autoscaleconstant(scaleboost=0.2)
    tree.autoscaleconstant(scaleboost=0.2)
    if iLoop%3 == 0:
        tree.minexpand(scaleconst=1e0)
        tree.minexpand(scaleconst=1e-6)
    tree.setplayouts(10)
    expandhead_radiallatin(5, len(ubounds), tree, 1e-3)
#    expandhead_latin(5, tree, lbounds, ubounds)
    tree.setplayouts(20)
    tree.savetree(restartfile)
    minval = tree.getbestscore()
    if minval < lastmin:
        minval = min(lastmin, minval)
        lastloop = iLoop

    if minval < 1e-4:
        break

#    if iLoop-lastloop > 4:
#        tree.minexpand(scaleconst=1e-6, mindepth=3)
#        tree.graft()
#    tree.graft()
#    tree.savetree(restartfile)
 #   print("Best Score: %s"%(minval))

    if iLoop%4 == 10:
        tree.selectpath()
    print(tree)
    curtime = time()
    print("Search Duration: %s"%(curtime-starttime))
#    tree.setheadexpand(tree.getheadexpand() + 2)
#    tree.autoscaleconstant(scaleboost=1.0)
    newconst = tree.getconstant()
    print("Loop Number: %s, ExploreConst: %s"%(iLoop, newconst) )

