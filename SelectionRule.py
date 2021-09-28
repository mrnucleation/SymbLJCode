from math import exp, sqrt, log, fabs
from random import random, choice, seed, shuffle

#from ParameterObject import depthscale
#depthlimit = len(depthscale)+500000
#========================================================
def UBEnergy(nodelist, exploreconstant, verbose):
    #Compute the average uniqueness factor
    uniqscore = [1.0 for x in nodelist]
    for i, node in enumerate(nodelist):
        visits = node.getvisits()
        score = node.getuniquenessdata(nodelist)
        uniqscore[i] = sqrt(visits/score)
#    uniqscore[0] = 0.0

#    maxval = max(uniqscore)
#    try:
#        uniqscore = [1.5*x/maxval for x in uniqscore]
#    except ZeroDivisionError:
#        uniqscore = [0.0 for node in nodelist]
    keylist = {}
    for i, node in enumerate(nodelist):
        keylist[str(node)] = uniqscore[i]

    selection = sorted(nodelist, key=lambda x:x.getid())
    selection = sorted(selection, key=lambda x:UCT_Unique_Score(x, keylist[str(x)], exploreconstant, doprint=True))[-1]
    print("Selecting Node %s with Score: %s"%(selection.getid(),  UCT_Unique_Score(selection, keylist[str(selection)], exploreconstant, doprint=True)  ))
    return selection
#==========================================================

def UCT_Unique_Score(node, uniqval, exploreconstant, doprint=False):
    
    parent = node.getparent()
    energy = node.getscore()
    visits = node.getvisits()
#        nChildren = len(node.getchildren())
    if parent is None:
#        return -1e30
        parenergy = node.getscore()
        parvisits = visits
    else:
        parenergy = parent.getscore()
        parvisits = parent.getvisits()


    depth = node.getdepth()
#    _, playoutEList = node.getplayouts()
    _, playoutEList = node.getallplayouts()
#    usedlist = node.getusedlist()
#        playoutEList = node.getenergylist()
    childeng = [child.getscore() for child in node.getnodelist()]
    nodeEnergy = node.getscore()
    nodeweight = nodeEnergy
#    scalefunc = lambda x: log(x, 10.0)
#    scalefunc = lambda x: sqrt(x)
    scalefunc = lambda x: x
    if visits < 10:
        exploitweight = scalefunc(nodeweight)
    else:
        exploitweight = 1e300
    cnt = 1
#        if len(childeng) > 0:
#            for energy in childeng:
#                exploitweight = min(exploitweight, scalefunc(energy))
#                exploitweight += energy
#                cnt += 1
    if len(playoutEList) > 0:
        for i, energy in enumerate(playoutEList):
            exploitweight = min(exploitweight, scalefunc(energy))
#                exploitweight += energy
#                cnt += 1
    exploitweight = exploitweight/cnt
    explore = 0.0
    try:
        explore = uniqval*sqrt(log(parvisits)/visits)
    except (ValueError, ZeroDivisionError):
        explore = uniqval

    score = -exploitweight + exploreconstant*explore
    if parent is not None:
        node.setexploitvalue(-exploitweight)
        node.setexplorevalue(explore)
    if depth > depthlimit or (parent is None):
        if doprint:
            try:
                print("Node %s (Parent:%s, Depth %s, Visits:%s): Exploit: %s Score:%s"%(node.getid(), parent.getid(), depth, visits, -exploitweight, -1e20))
            except:
                print("Node %s (Parent:Head, Depth %s, Visits:%s): Exploit: %s Score:%s"%(node.getid(), depth, visits, -exploitweight, -1e20))
        return -1e20


    if doprint:
        if parent is None:
            print("Node %s (Parent:%s, Depth:%s, Visits:%s): Exploit:%s  Explore:%s Score:%s"%(node.getid(), 'Head', depth, visits, -exploitweight, exploreconstant*explore, score))
        else:
            print("Node %s (Parent:%s, Depth:%s, Visits:%s): Exploit:%s  Explore:%s Unique:%s Score:%s"%(node.getid(), parent.getid(), depth, visits, -exploitweight, exploreconstant*explore, uniqval, score))
    return score



#========================================================

