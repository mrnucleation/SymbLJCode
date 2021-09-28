from math import exp, sqrt, log, fabs
from random import random, choice, seed, shuffle
#========================================================
def UBUnique(nodelist, exploreconstant, verbose=False):
    '''
    Default selection rule with uniqueness criteria. 
    '''
    #Compute the average uniqueness factor
    uniqscore = [1.0 for x in nodelist]
#    for i, node in enumerate(nodelist):
#        score = node.getuniquenessdata(nodelist)
#        uniqscore[i] = 1.0/score
#    maxscore = max(uniqscore)
#    uniqscore = [x/maxscore for x in uniqscore]

    keylist = {}
    for i, node in enumerate(nodelist):
        keylist[str(node)] = uniqscore[i]

    selection = sorted(nodelist, key=lambda x:x.getid())
    selection = sorted(selection, key=lambda x:UCT_Unique_Score(x, keylist[str(x)], exploreconstant, doprint=verbose))[-1]
    print("Selecting Node %s with Score: %s"%(selection.getid(),  UCT_Unique_Score(selection, keylist[str(selection)], exploreconstant, doprint=verbose)  ))
    return selection
#==========================================================

def UCT_Unique_Score(node, uniqval, exploreconstant, doprint=False):
    
    parent = node.getparent()
    energy = node.getscore()
    visits = node.getvisits()
    if parent is None:
        parenergy = node.getscore()
        parvisits = visits
    else:
        parenergy = parent.getscore()
        parvisits = parent.getvisits()


    depth = node.getdepth()
    _, playoutEList = node.getallplayouts()
    usedlist = node.getusedlist()

    childeng = [child.getscore() for child in node.getnodelist()]
    nodeEnergy = node.getscore()
    nodeweight = nodeEnergy

    scalefunc = lambda x: x
    exploitweight = nodeEnergy
#    if len(playoutEList) < 30:
#        exploitweight = nodeEnergy
#    else:
#        exploitweight = 1e300
    cnt = 1

    if len(playoutEList) > 0:
        for i, energy in enumerate(playoutEList):
            if i not in usedlist:
                exploitweight = min(exploitweight, scalefunc(energy))
    exploitweight = exploitweight/cnt
    if exploitweight > 1e4:
        exploitweight = 1e4
    explore = 0.0
    try:
        explore = uniqval*sqrt(log(parvisits)/visits)
    except (ValueError, ZeroDivisionError):
        explore = uniqval

    score = -exploitweight + exploreconstant*explore
    if parent is not None:
        node.setexploitvalue(-exploitweight)
        node.setexplorevalue(explore)
#    if node.isminned() or (parent is None):
    if parent is None:
        if doprint:
            try:
                print("Node %s (Parent:%s, Depth %s, Visits:%s): Exploit: %.4e Score:%.4e"%(node.getid(), parent.getid(), depth, visits, -exploitweight, -1e20))
            except:
                print("Node %s (Parent:Head, Depth %s, Visits:%s): Exploit: %.4e Score:%.4e"%(node.getid(), depth, visits, -exploitweight, -1e20))
        return -1e20


    if doprint:
        if parent is None:
            print("Node %s (Parent:%s, Depth:%s, Visits:%s): Exploit:%.4e  Explore:%.4e Score:%.4e"%(node.getid(), 'Head', depth, visits, -exploitweight, exploreconstant*explore, score))
        else:
            print("Node %s (Parent:%s, Depth:%s, Visits:%s): Exploit:%.4e  Explore:%.4e Unique:%.4e Score:%.4e"%(node.getid(), parent.getid(), depth, visits, -exploitweight, exploreconstant*explore, uniqval, score))
    return score
#========================================================

