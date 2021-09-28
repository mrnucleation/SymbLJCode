import os
import natsort
from glob import glob
from random import random, shuffle
from time import time
from vaspUtilities import getTotalEnergy, getForces, getPositions, getPosForces
from EAMCalc import eamenergies, readdump

import numpy as np
import warnings
import functions
from math import fabs
from FF_Model import ForcefieldModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
activation_funcs = {
    'pair':
    [[
#      *[functions.Constant()] * 1,
      *[functions.Identity()] * 1,
      *[functions.Square()] * 1,
      *[functions.Pow(power=-12.0)] * 1,
      *[functions.Pow(power=-9.0)] * 1,
      *[functions.Pow(power=-6.0)] * 1,
      *[functions.Pow(power=-3.0)] * 1,
      *[functions.Pow(power=-1.0)] * 1,
    ],
    [
#      *[functions.Constant()] * 1,
      *[functions.Identity()] * 1,
#      *[functions.Square()] * 1,
#      *[functions.Sin()] * 2,
#      *[functions.Cos()] * 2,      
#      *[functions.Sigmoid()] * 1,
    ]],
}



zonewidth = 0.05
xhigh = zonewidth/2.0
xlow = -zonewidth/2.0


print("Using a Deadzone of width: %s"%(zonewidth))
print("Deadzone low/high: %s, %s"%(xlow, xhigh))
def deadzone(par, xlo, xhi):
    if par >= xhi:
        return par-xhi, 1
    elif par <= xlo:
        return par-xlo, 1
    else:
        return 0.0, 0

def inv_deadzone(par, xlo, xhi):
    if par >= 0:
        return par+xhi
    elif par <= 0:
        return par+xlo
    else:
        return 0.0

def lj(dist):
    r6 = np.power(dist, -6.0)
    r12 = np.power(dist, -12.0)
    eng = 4.0*(r12-r6)/2.0
    return eng
RCUT = 132.0

workdir = os.getcwd() + "/"

if os.path.exists(workdir+"LJ_TestData.npy"):
    testfusedstruct = np.load("LJ_TestData.npy")
    testexactlist = np.load("LJ_TestEnergies.npy")
#    testatomcount = np.load("LJ_TestSizes.npy")
    fusedstruct = np.load("LJ_TrainData.npy")
    exactlist = np.load("LJ_TrainEnergies.npy")
#    atomcount = np.load("LJ_TrainSizes.npy")
    ntrainstruct = exactlist.size
    nteststruct = testexactlist.size
    ntotalstruct = ntrainstruct + nteststruct
#    maxsize = max(atomcount.max(), testatomcount.max())
    maxsize = 2
    dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs)
else:


    maxsize = 2
#    dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
    dummymodel = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs)

    ntotalstruct = 1000
    nteststruct = round(0.2*ntotalstruct)
    ntrainstruct = ntotalstruct-nteststruct

    fusedstruct = np.random.uniform(0.4, 15.0, ntrainstruct)
    testfusedstruct = np.random.uniform(0.4, 15.0, nteststruct)
    fusedstruct = fusedstruct.astype(np.float32)
    testfusedstruct = testfusedstruct.astype(np.float32)
    exactlist = lj(fusedstruct)
    testexactlist = lj(testfusedstruct)

    #Process the input data into
    print("Train Set Size:%s"%(ntrainstruct))
    print("Train Set Size:%s"%(nteststruct))
    np.save("LJ_TestData", testfusedstruct)
    np.save("LJ_TestEnergies", testexactlist)
    np.save("LJ_TrainData", fusedstruct)
    np.save("LJ_TrainEnergies", exactlist)

frac_train = ntrainstruct/float(ntotalstruct)
frac_test = nteststruct/float(ntotalstruct)

curweight = dummymodel.get_npweights()
nParameters = 0
for i, row in enumerate(curweight):
    nParameters += row.size
print("Number of Parameters in the Model: %s"%(nParameters))

print(exactlist.shape)
print(fusedstruct.shape)
print(testexactlist.shape)
print(testfusedstruct.shape)

exactlist = tf.Variable(exactlist)
fusedstruct = tf.Variable(fusedstruct)
testexactlist = tf.Variable(testexactlist)
testfusedstruct = tf.Variable(testfusedstruct)


#========================================================
def nplossfunc(y_predict, y_target):
    if np.any(tf.math.is_nan(y_predict)):
        return 1e18
    err = y_predict - y_target
    errmax = tf.math.reduce_max(err)
    err = tf.math.subtract(err, errmax)
    score = tf.math.square(err)
    score = tf.math.reduce_mean(score)
    score = score.numpy()
    return score
#========================================================
def rmse(y_predict, y_target):
    if np.any(tf.math.is_nan(y_predict)):
        return 1e18
    err = tf.math.subtract(y_predict, y_target)
    err = tf.math.square(err)
    err = tf.math.reduce_mean(err)
    score = tf.math.sqrt(err)
    return score.numpy()
#========================================================
def mae(y_predict, y_target, numpyout=True):
    if np.any(tf.math.is_nan(y_predict)):
        if numpyout:
            return 1e18
        else:
            print(y_predict)
            return tf.constant(1e18)
    err = y_predict - y_target
    err = tf.math.abs(err)
    score = tf.math.reduce_mean(err)
    if numpyout:
        return score.numpy()
    else:
        return score
#========================================================
def runsim(parameters, verbose=False, usemask=True):
    starttime = time()
    model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
    if usemask:
        maskedweights = [deadzone(x, xlow, xhigh) for x in parameters]
#        maskedweights = [0.0 if fabs(x) < 1e-1 else x for x in parameters]
        maskedweights, parmask = list(map(list, zip(*maskedweights)))
        maskcnt = sum(parmask)
    else:
        maskedweights = parameters
        maskcnt = 0

    curweight = model.get_npweights()
    cnt = -1
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            cnt += 1
            curweight[i][j] = maskedweights[cnt]
    model.set_npweights(curweight)

    if verbose:
        model.pretty_output()
        model.create_plot()
    cnt = 0

        
    with warnings.catch_warnings():    
        warnings.filterwarnings('ignore', r'overflow encountered in')
        warnings.filterwarnings('ignore', r'overflow encountered in reduce')
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
        warnings.filterwarnings('ignore', r'Input contains NaN')

        with tf.device('/CPU:0'):
            result = model(fusedstruct)
            maescore = mae(result, exactlist)
            rmsescore = rmse(result, exactlist)
            if np.all(np.where(np.abs(result.numpy()) > 1e-52)):
                score = 1e18
            else:
                score = maescore
#                score = rmsescore
#            score = nplossfunc(result, exactlist)
            if verbose:
#                maescore = mae(result, exactlist)
                print("Train MAE: %s, Train RMSE: %s"%(maescore, rmsescore))
#            score = maescore

    if verbose:
        with open("corr_train.dat", "w") as outfile:
            for e_trial, e_ref in zip(result.numpy(), exactlist.numpy()):
                outfile.write("%s %s\n"%(e_trial, e_ref))

    if np.isinf(score) or np.isnan(score):
        score = 1e18

    if score > 1e18:
        score = 1e18
        testscore = 1e18
    else:
        with tf.device('/CPU:0'):
            testresult = model(testfusedstruct)
            maescore = mae(testresult, testexactlist)
            rmsescore = rmse(testresult, testexactlist)
            if np.all(np.where(np.abs(testresult.numpy()) > 1e-52)):
                testscore = 1e18
            else:
                testscore = maescore
#                testscore = rmsescore
#            testscore = nplossfunc(testresult, testexactlist)
            if verbose:
#                maescore = mae(testresult, testexactlist)
                print("Test MAE: %s, Test RMSE: %s"%(maescore, rmsescore))
#            testscore = maescore
    if np.isinf(testscore) or np.isnan(testscore):
        testscore = 1e18

    if testscore > 1e18:
        testscore = 1e18

    if verbose:
        with open("corr_test.dat", "w") as outfile:
            for e_trial, e_ref in zip(testresult.numpy(), testexactlist.numpy()):
                outfile.write("%s %s\n"%(e_trial, e_ref))

    totalscore = frac_test*testscore + frac_train*score

    endtime = time()
    tf.keras.backend.clear_session()
    print("Score: %s, Runtime: %s"%(score, endtime-starttime))
    if not verbose:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s | %s | %s  \n'%(outstr, testscore, score, totalscore))
        with open("dumpfile_mask.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in maskedweights])
            outfile.write('%s | %s | %s | %s  \n'%(outstr, testscore, score, totalscore))

    else:
        print("Test Score:%s"%(testscore))
    return score
#========================================================
def tf_minimize(parameters, nepoch = 600, usemask=True):
    starttime = time()
    model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs, maxatoms=maxsize)
    if usemask:
        maskedweights = [deadzone(x, xlow, xhigh) for x in parameters]
        maskedweights, parmask = list(map(list, zip(*maskedweights)))
        maskcnt = sum(parmask)
    else:
        maskedweights = parameters
        maskcnt = 0
        parmask = []
#    print(maskedweights)

    startscore = runsim(maskedweights, usemask=False)

    curweight = model.get_npweights()
    tfmask = model.get_npweights()
    cnt = -1
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            cnt += 1
            curweight[i][j] = maskedweights[cnt]
            if usemask:
                tfmask[i][j] = round(float(parmask[cnt]))
    if usemask:
        for i, layer in enumerate(tfmask):
            tfmask[i] = tf.constant(layer)


    model.set_npweights(curweight)
    optimizer = keras.optimizers.Adam(learning_rate=2.5e-3)
    epochs = nepoch
    trainweights = model.get_weights()
    for epoch in range(epochs):
        # Iterate over the batches of the dataset.
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        timestart = time()
        with tf.device('/CPU:0'):
            with tf.GradientTape() as tape:
                tape.watch(trainweights)
                result = model(fusedstruct)
                loss_value = mae(result, exactlist, numpyout=False)
            grads = tape.gradient(loss_value, trainweights)
            for i, layer in enumerate(zip(grads, trainweights, tfmask)):
                glayer, wlayer, mlayer = layer
                glayer = tf.math.multiply(glayer, mlayer)
                grads[i] = tf.where(tf.math.is_nan(glayer), 0.0, glayer)
            optimizer.apply_gradients(zip(grads, trainweights))
        timeend = time()
        print("Training loss at step %d: %.4f, time:%s"
                    % (epoch, float(loss_value),timeend-timestart)
                )


    curweight = model.get_npweights()
    cnt = -1
    endparameters = []
    endmasked = []
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            endparameters.append(curweight[i][j])
            endmasked.append(inv_deadzone(curweight[i][j], xlow, xhigh))

    endmasked = np.array(endmasked)
    endmasked = np.where(endmasked == 0.0, parameters, endmasked)
    score = runsim(endmasked, usemask=True)
    endtime = time()
    return score, endmasked
#========================================================
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    par = []
    with open(filename, 'r') as parfile:
        for line in parfile:
            newpar = float(line.split()[0])
            par.append(newpar)
    score = runsim(par, verbose=True, usemask=False)


