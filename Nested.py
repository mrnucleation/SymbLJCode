from EAMCalc import eamenergies, writedump
import numpy as np
import os

from FF_Model import ForcefieldModel
from NNetCalc import activation_funcs, RCUT, activation_funcs, maxsize
from random import randint
from igraph import Graph
import tensorflow as tf
from math import isnan, fabs
from datetime import datetime
from time import time

#========================================================
def nestedrun(ffmodel):
    np.random.seed()
    #Initialize the 
#    print(structdir)
    simsize = 2
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir)
#    print(dumpdir)
    struct = np.zeros(shape=(simsize, 3))
    for i in range(struct.shape[0]):
        struct[i,0] = i*2.2


    distlist = ffmodel.buildrawdistlist(struct)
    atomcnt = np.array([simsize])
    elast = ffmodel(distlist, atomcnt).numpy()[0]
    print(atomcnt)
    if not clustcheck(distlist, simsize):
        raise Exception("Initial Cluster Criteria not met!")

    rmax = 1.0
    ndigits = 6
    dE = 10**(-ndigits)

    eMedian = 1e30

    emin = 1e300
    minstruct = np.copy(struct)
    filenumber = -1


    for iCycle in range(1,21):
        hist = {}
        norm = 0.0

        nacc = 0.0
        ntries = 0.0
        starttime = time()
#        if iCycle > 0:
#            newstruct = minstruct
#            elast = emin
        for iMove in range(50000):
            dX = np.random.uniform(-rmax, rmax, size=(3))
            trialAtom = np.random.randint(0, struct.shape[0])
            newstruct = np.copy(struct)
            newstruct[trialAtom, :] = newstruct[trialAtom, :] + dX[:]
            distlist = ffmodel.buildrawdistlist(newstruct)
            if mindistcheck(distlist, simsize):
                if clustcheck(distlist, simsize):
                    distlist = np.where(distlist > 50.0, 0.0, distlist)
                    energy = ffmodel(distlist, atomcnt).numpy()[0]
#                    print(energy)
                    if not isnan(energy):
                        if energy < eMedian:
                            elast = energy
                            struct = newstruct
                            nacc += 1.0
                            if energy < emin:
                                emin = energy
                                minstruct = np.copy(newstruct)
            eBin = round(round(elast/dE)*dE, ndigits)
            if eBin in hist:
                hist[eBin] += 1.0
            else:
                hist[eBin] = 1.0
            ntries += 1.0
            if iMove%100 ==0:
                if nacc/ntries < 0.5:
                    rmax *= 0.99
                else:
                    rmax *= 1.01
                rmax = max(rmax, 0.001)
        endtime = time()
        cycletime = endtime-starttime
        eMedian = computemedian(hist, ntries)
        outname = "cycle%s.data"%(iCycle)
        eEAM = eamenergies(struct=struct, outdatafile=outname)

        if fabs(eEAM-elast) > 10e-3:
            while True:
                filenumber += 1
                outfile = dumpdir+"neststruct%s.geo"%(filenumber)
                print("Writing %s"%outfile)
                if not os.path.exists(outfile):
                    if nacc >= 1.0:
                        writedump(struct, outfile)
                    break
                


        print("Cycle %s Time: %s, Model Median:%s, rMax:%s"%(iCycle, cycletime, eMedian, rmax))
        print("   Structure Candidate E_Model vs E_EAM :%s, %s"%(elast, eEAM))
        print("   Acceptance Rate :%s"%(1e2*nacc/ntries))
        if nacc < 1.0:
            break



#========================================================
def mindistcheck(distlist, nAtoms):
    mindist = np.where((1e-100 < distlist) & (distlist < 1.25))
    if np.any(mindist):
#        print(distlist)
        return False
    else:
        return True
#========================================================
def clustcheck(distlist, nAtoms):
    graph = Graph()
    if nAtoms == 1:
        return True
    elif nAtoms == 2:
        if distlist[0,1] < 7.0:
            return True
        else:
            return False
    for iAtom in range(nAtoms):
        graph.add_vertices(iAtom)
    for iAtom in range(nAtoms):
        for jAtom in range(iAtom+1, nAtoms):
            if (distlist[iAtom, jAtom] > 1e-50) and (distlist[iAtom, jAtom] < 7.0):
                graph.add_edges( [(iAtom, jAtom)] ) 
    cluster = graph.clusters()[0]

    if len(cluster) < nAtoms:
        return False
    else:
        return True
#========================================================
def computemedian(hist, norm):
    sortedhist = [ (energy, cnt) for energy, cnt in hist.items() ]
    sortedhist = sorted( sortedhist, key=lambda x:x[0])
    integral = 0.0
#    print(sortedhist)
    for iBin, data in enumerate(sortedhist):
        energy, cnt = data
        integral += cnt
        median = energy
        if integral > 0.5*norm:
            lastdata = sortedhist[iBin-1]
            lasteng, lastcnt = lastdata
            median = 0.5*(energy+lasteng)
#            print(iBin, energy)
            break
    return median
#========================================================
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    par = []
    with open(filename, 'r') as parfile:
        for line in parfile:
            newpar = float(line.split()[0])
            par.append(newpar)
    model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)

    curweight = model.get_npweights()
    cnt = -1
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            cnt += 1
            curweight[i][j] = par[cnt]
    model.set_npweights(curweight)
    nestedrun(model)
