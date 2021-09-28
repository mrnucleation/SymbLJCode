from random import random, shuffle
from time import time
from copy import deepcopy
from math import sqrt
from NNetCalc import tf_minimize
try:
    from hyperopt import hp, fmin, tpe, space_eval, Trials
except:
    pass
import sys
import numpy as np
#================================================
class MissingParameters(Exception):
    pass
#================================================
class InvalidParameterBounds(Exception):
    pass
#================================================
class ParameterData(object):
    '''
     Default Hypersphere Search for a cMCTS algorithm. 
    '''
    #------------------------------------------------
    def __init__(self, parameters, lossfunction, lbounds, ubounds, depthscale=None, options={}):
        '''
         Input Variables:
           parameters => List or numpy array of the trial parameters for this data obbject
           lossfunction => C
           lbounds => Lower bounds for the parameter search. 
           ubounds => Upper bounds for the parameter search. 
           depthscale => The maximum distance that rollouts and child nodes are allowed
           options => Additional non-manditory control values that can be set.
        '''
        self.options = options
        self.parameters = parameters
        if len(self.parameters) < 1:
            raise MissingParameters("Please specifiy an initial parameter set")
        self.dimensions = len(self.parameters)
        self.parameters = np.array(self.parameters)
        self.score = 0.0

        if depthscale is None:
            self.depthscale = [1.0, 0.25, 0.2, 0.15, 0.1]
            self.depthscale = [sqrt(self.dimensions)*x for x in self.depthscale]
        else:
            self.depthscale = depthscale

        if (lbounds is None) or (ubounds is None):
            raise InvalidParameterBounds("The upper and lower parameter bounds must be defined!")
        self.childlb = None
        self.childub = None

        if 'customplayout' in options:
            self.customplayout = options['boundaryscale']
            if not callable(self.customplayout):
                raise TypeError("A custom playout must be a callable function with a call signature f(dataobj, nplayouts, node)")
        else:
            self.customplayout = None

        if not callable(lossfunction):
            raise TypeError("The objective/loss function must be a callable function which takes the current parameter set as an argument!")
        self.lossfunction = lossfunction

        self.comparedlist = {}
        self.rcount = 0


        if 'boundaryscale' in options:
            self.boundaryscale = options['boundaryscale']
        else:
            self.boundaryscale = True

        if 'verbose' in options:
            if options['verbose'] == 2:
                self.verbose = True
            else:
                self.verbose = False
        else:
            self.verbose = False


        # This portion checks the consistency of the number of bounds and the parameters
        # to make sure the user properly defines their optimization problem.
        if len(lbounds) != len(ubounds):
            raise InvalidParameterBounds("The number of upper and lower bounds do not match!")
        if len(lbounds) != self.dimensions:
            raise InvalidParameterBounds("The number of bounds given do not match the number of parameters!")
        for lower, upper in zip(lbounds, ubounds):
            if upper <= lower:
                print(lower, upper)
                raise InvalidParameterBounds("Lower Bound >= Upper Bound!")

        self.lbounds = np.array(lbounds)
        self.ubounds = np.array(ubounds)
        self.bay_trials = None


    #----------------------------------------------------
    def cleanup(self, node):
        return

    #----------------------------------------------------
    def __eq__(self, dataobj2):
        str2 = dataobj2.getstructure()
        return str2 == self.parameters
    #----------------------------------------------------
    def __str__(self):
        outstr = ' '.join([str(x) for x in list(self.parameters)])
        return outstr
    #----------------------------------------------------
#    def computechildbounds(self, node):
#        searchmax = self.getsearchmax(node)
#        if self.childlb is None:
#            self.childlb = []
#            self.childub = []
#            for par, lb, ub in zip(self.parameters, self.lbounds, self.ubounds):
#                upper = par + searchmax*(ub-lb)
#                if upper > ub:
#                    upper = ub
#                lower = par - searchmax*(ub-lb)
#                if lower < lb:
#                    lower = lb
#                self.childlb.append(lower)
#                self.childub.append(upper)
#            self.childlb = np.array(self.childlb)
#            self.childub = np.array(self.childub)
    #----------------------------------------------------
    def computechildbounds(self, node):
        searchmax = self.getsearchmax(node)
        if self.childlb is None:
            upper = self.parameters + searchmax*(self.ubounds-self.lbounds)
            upper = np.where(upper > self.ubounds, self.ubounds, upper)
            lower = self.parameters - searchmax*(self.ubounds-self.lbounds)
            lower = np.where(lower < self.lbounds, self.lbounds, lower)
            self.childub = upper
            self.childlb = lower
    #------------------------------------------------
    def setdepthscale(self, newdepthscale):
        self.depthscale = newdepthscale
    #------------------------------------------------
    def getdepthscale(self):
        return self.depthscale 
    #------------------------------------------------
    def getsearchmax(self, node):
        if node is not None:
            depth = node.getdepth()
            if depth < len(self.depthscale):
                searchmax = self.depthscale[depth]
            else:
                searchmax = self.depthscale[-1]
        else:
            depth = 0
            searchmax = self.depthscale[depth]
        return searchmax
    #------------------------------------------------
    def reducecoords(self, x, lb, ub):
        '''
         Returns the given coordinate set in reduced coordinates.  In this space x' = 0 
         corresponds to x = lb and x' = 1 corresponds to x = ub
        '''
        return np.divide(np.subtract(x,lb), np.subtract(ub,lb))
    #------------------------------------------------
    def newdataobject(self):
        newobj = ParameterData(parameters=self.parameters, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                              depthscale=self.depthscale, options=self.options)
        return newobj

    #------------------------------------------------
    def perturbate(self, node=None, parOnly=False):
        '''
         Function that when called produces a new parameter set by taking this data object's parameters
         and displacing it.
        '''
        self.computechildbounds(node)
        newlist = None
        while newlist is None:
            newlist = self.localshift(node=node)
        if not parOnly:
            newobj = ParameterData(parameters=newlist, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                                   depthscale=self.depthscale, options=self.options)
            return newobj
        else:
            return newlist
    #------------------------------------------------
    def localshift(self, node=None):
        depth = node.getdepth()
        if depth < 1:
            newlist = np.random.uniform(self.lbounds, self.ubounds)
            return newlist
        searchmax = self.getsearchmax(node)

        # We first need to compute the reduced array which maps the real parameter bounds
        # on to the interval (0,1) where 0 corresponds to the lower bound of each parameter
        # and 1 corresponds to the upper bound.
        x_r = np.divide( (self.parameters-self.lbounds), (self.ubounds-self.lbounds))
#        for i,par in enumerate(self.parameters):
#            x_r[i] = (par-self.lbounds[i])/(self.ubounds[i]-self.lbounds[i])

        #To generate on a unit sphere of N dimensions the quickest method is to create an
        #array of normally distributed random numbers and then scale the norm such that r=1
        #for the entire vector. This creates a point on the surface of a sphere with r=1.
        #We can then pick an r randomly which generates points randomly
        #on a unit sphere.
        u = np.random.normal(0.0, 1.0, self.dimensions)  
        norm = np.sum(u**2) **(0.5)

        #In the situation where the initial point is close to the boundary, using rmax as our
        #largest possible r value may not be possible since it would take us out of the bounded area.
        #As such we need to estimate how far we can go before hitting the boundary. 
        if self.boundaryscale:
            disttoedge = 1e300
            for x_old, dx in zip(x_r, u):
                if dx > 0.0:
                    disttoedge = min(disttoedge, 1.0-x_old)
                else:
                    disttoedge = min(disttoedge, x_old)
            rmax = min(disttoedge+1e-1, searchmax)
        else:
            rmax = searchmax

        r = random() * rmax
        x = r*u/norm
        x_r += x

        #This is a bounds check to ensure we didn't step outside of the bounds.
        x_r = np.where(x_r > 1.0, 1.0, x_r)
        x_r = np.where(x_r < 0.0, 0.0, x_r)
        newlist = np.multiply(self.ubounds-self.lbounds, x_r) + self.lbounds
        return newlist
    #----------------------------------------------------
    def crossover(self,  pardata, node=None, parnode=None):
        parlist = pardata.getstructure()

        self.computechildbounds(node)
        if node is not None and parnode is not None:
            if parnode.getid() == node.getid():
                raise
        score1 = self.getscore()
        score2 = pardata.getscore()

        minscore = min(score1, score2) - 1.0
        score1 = score1-minscore
        score2 = score2-minscore

        prob = score2/( score1 + score2 )
        newlist = []
        for mypar, theirpar in zip(self.parameters, parlist):
            if random() < prob:
                newlist.append(mypar)
            else:
                newlist.append(theirpar)
        newobj = ParameterData(parameters=newlist, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                              depthscale=self.depthscale, options=self.options)
        return newobj
    #----------------------------------------------------
    def runsim(self, playouts=1, node=None):
        '''
         Runs the playouts for a given node. 
        '''
        if self.customplayout  is None: 
            self.computechildbounds(node)
            _, playoutEList = node.getplayouts()
            nPrev = len(playoutEList)
            structlist = []
            if self.verbose:
                if node is not None:
                    print("Node Depth: %s"%(node.getdepth()))

            for playout in range(playouts):
                newlist = self.parameters
                tempstr = None
                while tempstr is None:
                    tempstr = self.perturbate(parOnly=True, node=node)
                newlist = tempstr
                structlist.append(newlist)
        else:
            structlist = self.customplayout(self, playouts, node)

        energylist = []
        for struct in structlist:
            energy = self.lossfunction(struct)
            energylist.append(energy)

        if self.verbose:
            for playout, energy in enumerate(energylist):
                    print("Playout %s Result: %s"%(nPrev+playout+1, energy))

        return energylist, structlist
    #------------------------------------------------------------------------------
    def runsim_bayesian(self, playouts=1, node=None):
        tries = 0
        structlist = []
        energylist = []

        searchmax = self.getsearchmax(node)
        space = []
        for i, par in enumerate(self.parameters):
            upper = par + searchmax*(self.ubounds[i]-self.lbounds[i])
            if upper > self.ubounds[i]:
                upper = self.ubounds[i]
            lower = par - searchmax*(self.ubounds[i]-self.lbounds[i])
            if lower < self.lbounds[i]:
                lower = self.lbounds[i]
            space.append(hp.uniform(str(i), lower, upper))

        if self.bay_trials is None:
            trials = Trials()
            nprev = 0
        else:
            trials = self.bay_trials
            nprev = len(trials.trials)


        try:
            results = fmin(self.lossfunction, space=space, algo=tpe.suggest, max_evals=nprev+playouts, trials=trials, verbose=0)
        except:
            pass
        self.bay_trials = trials
        cnt = nprev
#        print(len(trials.trials))
#        print(len(trials.results))

        for trial, result in zip(trials.trials[nprev:], trials.results[nprev:]):
            newlist = [trial['misc']['vals'][str(i)][0] for i in range(len(self.parameters))]
            energy = result['loss']
            cnt += 1
            if verbose:
                print("Playout %s Result: %s"%(cnt, energy))
            structlist.append(newlist)
            energylist.append(energy)
        newlist = [results[str(x)] for x in range(self.dimensions)]
#        for energy, struct in zip(energylist, structlist):
#            self.addhist(struct)
#            self.addscorehist(energy, struct)
        energy = min(trials.losses())
        return energylist, structlist     
    #----------------------------------------------------
    def findplayouts(self, node=None):
        structlist = []
        energylist = []

        return energylist, structlist
    #----------------------------------------------------
    def computescore(self, node=None):
        self.score = self.lossfunction(self.parameters)
        if self.verbose:
            print("Score: %s"%(self.score))
        return self.score
    #----------------------------------------------------
    def getscore(self):
        return self.score
    #----------------------------------------------------
    def setscore(self, score):
        self.score = score
    #----------------------------------------------------
    def getuniqueness(self, inlist=None, node=None, nodelist=None):
        if inlist is None:
            curlist = self.parameters
        else:
            curlist = inlist

        searchmax = self.getsearchmax(node)

        headnode = node.getlineage()[-1]
        playouts, _ = headnode.getallplayouts(addids=True)
        myplayouts = len(playouts)

        mypar = self.reducecoords(self.parameters, self.lbounds, self.ubounds)
        myid = node.getid()
        for playout in playouts:
            nodeid, playid, libstruct = playout
            if (nodeid,playid) in self.comparedlist:
                continue
#            r = np.sum(np.square(libstruct-curlist))
            r = np.linalg.norm(libstruct-curlist)
            if r < searchmax:
                self.rcount += 1
            self.comparedlist[(nodeid,playid)] = None
        score = (float(self.rcount)+myplayouts+1.0)
        return score
    #----------------------------------------------------
    def minimize(self, node=None):
        '''
         Runs a local gradient minimizer to push the solution to the final value. Usually called
         after the MCTS has narrowed the search space down to a small region.
        '''
        searchmax = self.getsearchmax(node)
        depth = node.getdepth()
        playstructs, playoutEList = node.getplayouts()
        initialguess = self.parameters
        minval = node.getscore()
        for struct, eng in zip(playstructs, playoutEList):
            if eng < minval:
                minval = eng
                initialguess = struct
#        epochscale = [100, 100, 200, 500, 700, 1000]
        epochscale = [200]
        try:
            nepoch = epochscale[depth]
        except IndexError:
            nepoch = epochscale[-1]

        self.computechildbounds(node)
        try:            
            score, newpar = tf_minimize(initialguess, nepoch=nepoch)
        except:
            score = minval
            newpar = initialguess
        if self.verbose:
            print("Pre-Minimization Score: %s"%(minval))
            print("Post Minimization Score: %s"%(score))
        newobj = self.newdataobject()  
        newobj.setstructure(newpar)
        return newobj, score
    #----------------------------------------------------
    def minimize_simplex(self, node=None):
        '''
         Runs a local gradient minimizer to push the solution to the final value. Usually called
         after the MCTS has narrowed the search space down to a small region.
        '''
#        return self.minimize_sgd(node=node)
#        return self.minimize_GA(node=node)
        searchmax = self.getsearchmax(node)
        playstructs, playoutEList = node.getplayouts()
        initialguess = self.parameters
        minval = node.getscore()
        for struct, eng in zip(playstructs, playoutEList):
            if eng < minval:
                minval = eng
                initialguess = struct

        self.computechildbounds(node)
        startstepsize=[(ub-lb)*searchmax for lb,ub in zip(self.lbounds, self.ubounds)]
        simplex = StephSimplex(self.lossfunction)
        results, score = simplex.runopt(self.lbounds, self.ubounds, initialguess=initialguess, maxeval=5000, 
                                        delmin=[1e-3 for x in self.lbounds])
        if self.verbose:
            print("Pre-Minimization Score: %s"%(self.score))
            print("Post Minimization Score: %s"%(score))
        newpar = [float(x) for x in results]
        newobj = ParameterData(parameters=newpar, lossfunction=self.lossfunction, lbounds=self.lbounds, ubounds=self.ubounds,
                              depthscale=self.depthscale, options=self.options)
        return newobj, score
    #----------------------------------------------------
    def minimize_GA(self, node=None):
        from GAOptimizer import GAOpt
        newobj, score = GAOpt(self, node=node, ngenerations=50)
        if self.verbose:
            print("Post Minimization Score: %s"%(score))
        return newobj, score
    #----------------------------------------------------
    def getstructure(self):
        return self.parameters
    #----------------------------------------------------
    def setstructure(self, parameters):
        self.parameters = np.array(parameters)
    #----------------------------------------------------
    def setlossfunction(self, evalfunc):
        self.lossfunction = evalfunc
    #----------------------------------------------------
    def convertstr(self, instr):
        '''
         Used to process input from the restartfile
        '''
        try:
            par = [float(x) for x in instr.strip().split()]
        except:
            newstr = instr.replace("[","")
            newstr = newstr.replace("]","")
            newstr = newstr.replace("\n","")
            newstr = newstr.strip()
            par = [float(x) for x in newstr.split(",")]
        par = np.array(par)
        return par
    #----------------------------------------------------
#================================================
