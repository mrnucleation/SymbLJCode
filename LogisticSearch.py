from sklearn.linear_model import LogisticRegression
import scipy.integrate as integrate
from scipy.optimize import minimize
from random import random, randint
from math import fabs
from copy import deepcopy

import sys
import numpy as np

from ParameterObject import ParameterData, InvalidParameterBounds, MissingParameters

#----------------------------------------------------------------
def plotrfuncs(rmodel, rmax):
    with open("plot.dat", "w") as outfile:
        for i in range(100):
            r = i*rmax/100.0
            outfile.write("%s %s\n"%(r/sqrt(nParameters), probr(r, rmodel, rmax)))

    totalarea, err = integrate.quad(probr, 0, rmax, args=(rmodel,rmax,))
    with open("intplot.dat", "w") as outfile:
        for i in range(100):
            r = i*rmax/100.0
            area, err = integrate.quad(probr, 0, r, args=(rmodel,rmax,))
            outfile.write("%s %s\n"%(r/sqrt(nParameters), area/totalarea))

#----------------------------------------------------------------
def polymodel(r, rmax):
    roots = r - np.arange(0.0, rmax*(101.1/100.0), rmax/100.0)
    roots = np.square(roots)
    roots = np.exp(-roots/1.0)
    return roots
#----------------------------------------------------------------
def probr(r, model, rmax):
    u = polymodel(r, rmax)
    prob = model.predict_proba(u.reshape(1,-1))[0][1]
    return prob
#----------------------------------------------------------------
def cdffunc(r, model, rmax):
    area, err = integrate.quad(probr, 0, r, args=(model,rmax,))
    return area
#----------------------------------------------------------------
def sampler(model, norm, rmax):
    prob = random()
    r0 = np.argmax(model.coef_[0])
    r0 = r0*rmax/100.0
    def obj(r):
        return (prob-cdffunc(r, model, rmax)/norm)**2
    startdiff = obj(r0)
    currmin = 0.0
    currmax = rmax
    # Since the CDF is a monotonically increasing function (IE if x1 > x2 then f(x1) > f(x2))
    # we can narrow down the value using a continuous version of a binary search.
    # This gives us a value that is close enough we can then minimize in just a few steps.
    for i in range(7):
#        r = (currmax-currmin)*random() + currmin
        r = (currmax+currmin)*0.5
        area = cdffunc(r, model, rmax)/norm
        curdiff = obj(r)
        if startdiff > curdiff:
            startdiff = curdiff
            r0 = r
        if area > prob:
            currmax = r
        if area < prob:
            currmin = r
    results = minimize(obj, r0, method='Nelder-Mead', tol=1e-4, options={'maxiter': 10000} )
#    results = minimize(obj, rmax*r0, method='CG', tol=1e-3, options={'maxiter': 50000} )
    r = results.x[0]
    if r > rmax:
        r = rmax
    elif r < 0.0:
        r = 0.0
    
    return r
#----------------------------------------------
def generatesign(model):
    nparameter = len(model.coef_[0])
    u = np.sign(model.coef_[0]) 
    probold = model.predict_proba(u.reshape(1,-1))[0][1]
    for i in range(nparameter//5):
        site = randint(0, nparameter-1)
        u_new = np.copy(u)
        u_new[site] *= -1.0 
        probnew = model.predict_proba(u_new.reshape(1,-1))[0][1]
        if probnew > probold:
            u = u_new
            probold = probnew
        elif probnew/probold > random():
            u = u_new
            probold = probnew
    return u
#================================================
class LogisticSearch(ParameterData):
    '''
     Biased Sampling approach using Logistic Regression to learn from previous trials 
     in order to predict which direction is likely to lower the loss function.
     Designed primarily for functions that are more convex in nature though can still be used
     for other functions. 
    '''
    #------------------------------------------------
    def __init__(self, parameters, lossfunction, lbounds, ubounds, depthscale=None, options={}, directmodel=None, distmodel=None):
        super().__init__(parameters, lossfunction, lbounds, ubounds, depthscale, options)
        self.evalsince = 0
        self.rnorm = None
        if directmodel is None:
            self.distancemodel = LogisticRegression( max_iter=10000 )
            self.directionmodel = LogisticRegression(max_iter=10000 )
            self.trained = False
            self.lasttrain = 0
            self.modelrmax = 0.0
        else:
            self.distancemodel = deepcopy(distmodel)
            self.directionmodel = deepcopy(directmodel)
            self.directionmodel.coef_
            self.trained = False
            self.lasttrain = 0
            self.modelrmax = 0.0

    #------------------------------------------------
    def newdataobject(self):
        if self.trained:
            newobj = LogisticSearch(parameters=self.parameters, 
                                    lbounds=self.lbounds, 
                                    ubounds=self.ubounds, 
                                    lossfunction=self.lossfunction, 
                                    directmodel=self.directionmodel, 
                                    distmodel=self.distancemodel, 
                                    options=self.options, 
                                    depthscale=self.depthscale)
        else:
            newobj = LogisticSearch(parameters=self.parameters, 
                                    lbounds=self.lbounds, 
                                    ubounds=self.ubounds, 
                                    lossfunction=self.lossfunction, 
                                    options=self.options,
                                    depthscale=self.depthscale)
        return newobj

    #------------------------------------------------
    def perturbate(self, node=None, parOnly=False, forceuniform=False, cursims=None, curtrials=None):
        self.computechildbounds(node)

        newlist = None
#        newlist = self.localshift(node=node)
        if cursims is None:
            while newlist is None:
                newlist = self.logistic_localshift(node=node, curtrials=curtrials)
        else:
            while newlist is None:
                newlist = self.intialize_localshift(node=node, cursims=cursims, curtrials=curtrials)


        selection = newlist
        if newlist is None:
            print("Unable to perform move!")
        if not parOnly:
            if self.trained:
                newobj = LogisticSearch(parameters=selection, 
                                        lbounds=self.lbounds, 
                                        ubounds=self.ubounds, 
                                        lossfunction=self.lossfunction, 
                                        directmodel=self.directionmodel, 
                                        distmodel=self.distancemodel, 
                                        depthscale=self.depthscale,
                                        options=self.options)
            else:
                newobj = LogisticSearch(parameters=selection, 
                                        lbounds=self.lbounds, 
                                        ubounds=self.ubounds, 
                                        lossfunction=self.lossfunction, 
                                        depthscale=self.depthscale,
                                        options=self.options)
            return newobj
        else:
            return selection
    #------------------------------------------------
    def intialize_localshift(self, node=None, cursims=0, curtrials=None):
        '''
         This subroutine is designed to generate the first 6 random trials of a new node
        '''
#        return self.localshift(node=node)
        searchmax = self.getsearchmax(node)
        depth = node.getdepth()
        if depth < 1:
            return self.localshift(node=node)

        newlist = [0.0 for x in self.parameters]
        x_r = self.reducecoords(self.parameters, self.lbounds, self.ubounds)

        _, playoutEList = node.getallplayouts()
        oldsims = len(playoutEList)

        #To help the logistic model out a set of preset vector directions are generated to 
        #test previousy successful directions.This direction was successful once so check it again.
        #
        #The first vector is created to be a continuation of the vector that was generated
        #from the parent node's paraeter set. 
        #
        #The second tries to take the parent node's logistic model and uses it to generate a sample
        #using those coefficients. This is to see if the parent's logistic model's predictions are still good
        #
        #The third and fourth generate vectors that point toward and way from the center of the hypercube.
        #
        #The fifth and sixth generate two vectors in thethe all positive and all negative directions. If the variables
        #are independent this should yield decrease in one of the directions unless the system is at a fixed point. 
        #
        if isinstance(cursims, int):
            oldsims += cursims
        if oldsims < 1:
            # Generates a random vector in the same direction the parent
            # took to get to this node's parameters. IE it's checking if the previously
            # successful direction is still successful.
            parent = node.getparent()
            if parent is None:
                return self.localshift(node=node)
            parentpar = parent.getdata().getstructure()
            parentpar = self.reducecoords(parentpar, self.lbounds, self.ubounds)
            x_r = self.reducecoords(self.parameters, self.lbounds, self.ubounds)
            u = x_r - parentpar
        elif oldsims < 2:
            # Generates a random vector in the same direction as the parent's logistic model.
            try:
                direction = np.sign(self.directionmodel.coef_[0]) 
            except:
                return self.localshift(node=node)
            u = np.abs(np.random.normal(0.0, 1.0, len(self.parameters)))
            u = np.multiply(u, direction)

        elif oldsims < 3:
            # Generates a random vector toward the box center
            x_r = self.reducecoords(self.parameters, self.lbounds, self.ubounds)
            boxcenter = self.reducecoords((self.ubounds+self.lbounds)*0.5, self.lbounds, self.ubounds)
            direction = x_r - boxcenter
            direction = np.sign(direction) 
            u = np.abs(np.random.normal(0.0, 1.0, len(self.parameters)))
            u = np.multiply(u, direction)

        elif oldsims < 4:
            # Generates a random vector away from the box center
            x_r = self.reducecoords(self.parameters, self.lbounds, self.ubounds)
            boxcenter = self.reducecoords((self.ubounds+self.lbounds)*0.5, self.lbounds, self.ubounds)
            direction = x_r - boxcenter
            direction = -np.sign(direction) 
            u = np.abs(np.random.normal(0.0, 1.0, len(self.parameters)))
            u = np.multiply(u, direction)

        elif oldsims < 5:
            # Generates a random vector in the positive direction
            u = np.full(len(self.parameters), 1.0) 
        elif oldsims < 6:
            # Generates a random vector in the negative direction
            u = np.full(len(self.parameters), -1.0)
            if self.verbose:
                print("Initial Trials Finished")
        else:
            #There's always a chance the model is awful.  As such
            #we stil want to mix in purely random playouts with the
            #logistic based ons in order to avoid trusting the model
            #too much. 
            if random() < 0.5:
                return self.logistic_localshift(node=node, curtrials=curtrials)
            else:
                return self.localshift(node=node)
        

        #If boundaryscale is set, this truncates the circle if the circle interscets with the 
        #boundary ensuring solutions that are well outside of the boundary are excluded.  
        #A small smudge factor is added to allow a non-zero chance of picking the boundary 
        #itself as a solution.  
        norm = np.linalg.norm(u)
        u = u/norm
        if self.boundaryscale:
            disttoedge = 1e300
            for x_old, dx in zip(x_r, u):
                if dx > 0.0:
                    disttoedge = min(disttoedge, 1.0-x_old)
                else:
                    disttoedge = min(disttoedge, x_old)
            rmax = min(disttoedge+5e-2, searchmax)
        else:
            rmax = searchmax
        r = random() * rmax
        x = r*u
        x_r += x

        #This is a bounds check to ensure we didn't step outside of the bounds.
        x_r = np.where(x_r > 1.0, 1.0, x_r)
        x_r = np.where(x_r < 0.0, 0.0, x_r)

        newlist = np.multiply(self.ubounds-self.lbounds, x_r) + self.lbounds
        searchmax = self.getsearchmax(node)

        return newlist

    #------------------------------------------------
    def logistic_localshift(self, node=None, curtrials=None):
        searchmax = self.getsearchmax(node)
        depth = node.getdepth()
        playoutStruct, playoutEList = node.getallplayouts()
        if curtrials is not None:
            curstruct, curEng = curtrials
            playoutStruct = playoutStruct + curstruct
            playoutEList = playoutEList + curEng

        if depth == 0:
            return self.localshift(node=node)

        x_r = self.reducecoords(self.parameters, self.lbounds, self.ubounds)

        rmax = searchmax
        self.evalsince += 1
        if self.evalsince > 5 or (not self.trained):
            X = []
            Y = []
            R = []
            W = []
            trainable = False
            lowerval = False
            upperval = False
            bestr = 0.0
            bestscore = self.score
            mypar_reduced = self.reducecoords(self.parameters, self.lbounds, self.ubounds)
            for par, score in zip(playoutStruct, playoutEList):
                par_reduced = self.reducecoords(par, self.lbounds, self.ubounds)
                x = par_reduced - mypar_reduced
                r = np.linalg.norm(x)
                x = np.sign(x)
                if score < self.score:
                    y = 1.0
#                    w = min(fabs(score - self.score)/fabs(self.score), 0.05)
                    w = 1.0
                    lowerval = True
                    if bestscore > score:
                        bestscore = score
                        bestr = r
                else:
                    y = 0.0
                    w = 1.0  
                    upperval = True
                W.append(w)
                X.append(x)
                Y.append(y)
                R.append(r)
            if lowerval and upperval:
                trainable = True
            if not trainable:
#                if self.verbose:
#                    print("Model Not Trainable...Defaulting")
                return self.localshift(node=node)
            if self.verbose:
                print("Retraining Node %s's Logistic Model"%(node.getid()))
            W = np.array(W)
            r_train = np.array([polymodel(r, searchmax) for r in R])
            try:
#                self.directionmodel.fit(X,Y,sample_weight = W)
#                self.distancemodel.fit(r_train,Y,sample_weight = W )
                self.directionmodel.fit(X,Y)
                self.distancemodel.fit(r_train,Y)
                if self.verbose:
                    print("Model Trained, Best R: %s, SearchMax: %s"%(bestr, searchmax))
            except ValueError as e:
                if self.verbose:
                    print("Model can't be currently trained. Defaulting..")
                    print(e)
                return self.localshift(node=node)
            self.rnorm, err = integrate.quad(probr, 0, searchmax, args=(self.distancemodel,searchmax,))
            self.trained = True
            self.lasttrain = len(playoutEList) 
            self.evalsince = 0

        if not self.trained:
            return self.localshift(node=node)
        if self.rnorm is None:
            self.rnorm, err = integrate.quad(probr, 0, searchmax, args=(self.distancemodel,searchmax,))
        direction = generatesign(self.directionmodel)
        if random() < 0.05:
            direction = direction*-1
        u = np.abs(np.random.normal(0.0, 1.0, len(self.parameters)))
        u = np.multiply(u, direction)

        if self.boundaryscale:
            disttoedge = 1e300
            for x_old, dx in zip(x_r, u):
                if dx > 0.0:
                    disttoedge = min(disttoedge, 1.0-x_old)
                else:
                    disttoedge = min(disttoedge, x_old)
            rmax = min(disttoedge+5e-2, searchmax)
        else:
            rmax = searchmax
        norm = np.sum(u**2) **(0.5)
        r = sampler(self.distancemodel, self.rnorm, searchmax)
        if r > searchmax:
            raise ValueError("Logistic Model Broke Distance Constraint!")
        x = r*u/norm
        x_r += x

        #This is a bounds check to ensure we didn't step outside of the bounds.
        x_r = np.where(x_r > 1.0, 1.0, x_r)
        x_r = np.where(x_r < 0.0, 0.0, x_r)
        newlist = np.multiply(self.ubounds-self.lbounds, x_r) + self.lbounds

        return newlist
    #----------------------------------------------------
    def runsim(self, playouts=1, node=None):
        '''
         Runs the playouts for a given node. 
        '''
        self.computechildbounds(node)
        _, playoutEList = node.getplayouts()
        nPrev = len(playoutEList)
        structlist = []
        energylist = []
        if self.verbose:
            if node is not None:
                print("Node Depth: %s"%(node.getdepth()))

        for playout in range(playouts):
            newlist = self.parameters
            tempstr = None
            while tempstr is None:
                tempstr = self.perturbate(parOnly=True, node=node, forceuniform=False, cursims=playout,
                                          curtrials=(structlist, energylist))

            newlist = tempstr
            structlist.append(newlist)
            energy = self.lossfunction(newlist)
            energylist.append(energy)


        if self.verbose:
            for playout, energy in enumerate(energylist):
                    print("Playout %s Result: %s"%(nPrev+playout+1, energy))

        return energylist, structlist

#================================================

