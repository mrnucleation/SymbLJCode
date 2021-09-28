import sys
sys.path.append('/home/SHARE/metastable/SymbNetTest_rohit')

import natsort
import os
from shutil import copytree
from glob import glob
#from ScriptTemplate import InputTemplate
from random import random
from math import fabs, sqrt, sin, pi, copysign
from time import time


from sklearn.model_selection import train_test_split
import numpy as np
from FF_EAM_Fusedcall_Model import ForcefieldModel

import tensorflow as tf
import warnings
import functions
from utils import pretty_print
import pickle
import random
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

activation_funcs = {
    'pair':
    [[
    *[functions.Constant()] * 3,
    *[functions.Identity()] * 1,
    *[functions.Square()] * 1,
    *[functions.NegExp()] * 1,
    *[functions.Pow(power=-1.0)] * 1,
    *[functions.Pow(power=-3.0)] * 1,
    ],
    [
    *[functions.Constant()] * 5,
    *[functions.Identity()] * 5,    
    *[functions.Square()] * 1,
    *[functions.Sin()] * 2,
    *[functions.Cos()] * 2,      
    *[functions.Identity()] * 2,
    *[functions.Sigmoid()] * 2,
    ]],
    
    'density':
    [[
    *[functions.Constant()] * 3,
    *[functions.Identity()] * 1,
    *[functions.Square()] * 1,
    *[functions.Sin()] * 1,
    *[functions.Cos()] * 1,
    *[functions.Pow(power=-1.0)] * 1,
    *[functions.Pow(power=-2.0)] * 1,
    *[functions.Sigmoid()] * 2,        
    ]],
    
    'embedding':
    [[
    *[functions.Constant()] * 3,
    *[functions.Identity()] * 1,
    *[functions.Square()] * 1,
    *[functions.Sin()] * 1,
    *[functions.Cos()] * 1,
    *[functions.Sigmoid()] * 2,
    *[functions.Pow(power=-1.0)] * 1,
    *[functions.Pow(power=-2.0)] * 1,
    ],
    [
    *[functions.Constant()] * 5,
    *[functions.Identity()] * 5,    
    *[functions.Square()] * 1,
    *[functions.Sin()] * 2,
    *[functions.Cos()] * 2,      
    *[functions.Identity()] * 2,
    *[functions.Sigmoid()] * 2,
    ]]
}

DATA_LOCATION = "../../Al/Al_EAM_processed.dat"
TEST_DATA_LOCATION = "../../Al/testdata/Al/Al_test_EAM_processed2.dat"
mc_run = '15'
SAVEDIR = 'fusedmodels/v%s/'%mc_run
print_exprs = False
#modelname = 'fusedmodels/v%s/symmodel_epoch431.ml'%mc_run

RCUT = 6
n_epochs = 1000


#@tf.function
def loss(model, X, y):
    result = model(X[0],X[1],X[2],X[3])
    #cost = tf.reduce_mean(tf.square(tf.subtract(y,result)))     # MSE
    cost = tf.reduce_mean(tf.math.abs(tf.subtract(y,result)))    # MAE
    return cost
#----------------------------------------------------------------------

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
    vars = [var for var in tape.watched_variables()]
  return vars, loss_value, tape.gradient(loss_value, vars)
#----------------------------------------------------------------------

def prepare_input_data(positionlist, distlists, atomwisedistlist,
                       test_positionlist, test_distlists, test_atomwisedistlist):
    
    natomlist = np.zeros(len(distlists))
    npairlist = np.zeros(len(distlists))

    test_natomlist = np.zeros(len(test_distlists))
    test_npairlist = np.zeros(len(test_distlists))

    for i, distlist in enumerate(distlists):
        npairlist[i] = len(distlist)
        natomlist[i] = len(positionlist[i])


    for i, distlist in enumerate(test_distlists):
        test_npairlist[i] = len(distlist)
        test_natomlist[i] = len(test_positionlist[i])

    npairlist = npairlist.astype(int)
    natomlist = natomlist.astype(int)

    test_npairlist = test_npairlist.astype(int)
    test_natomlist = test_natomlist.astype(int)

    atomwisepairlist = []
    for idx in range(len(atomwisedistlist)):    
        Rs = distlists[idx]
        inner_list = []
        for atom_idx, rs in enumerate(atomwisedistlist[idx]):
            inner_list.append(np.in1d(Rs, rs).nonzero()[0].tolist())
        atomwisepairlist.append(inner_list)
        
    test_atomwisepairlist = []
    for idx in range(len(test_atomwisedistlist)):    
        Rs = test_distlists[idx]
        inner_list = []
        for atom_idx, rs in enumerate(test_atomwisedistlist[idx]):
            inner_list.append(np.in1d(Rs, rs).nonzero()[0].tolist())
        test_atomwisepairlist.append(inner_list)        
    
    
    distlists = np.concatenate(distlists)
    test_distlists = np.concatenate(test_distlists)
    
    X = [npairlist, natomlist, atomwisepairlist, distlists]
    test_X = [test_npairlist, test_natomlist, test_atomwisepairlist, test_distlists]
    
    return X, test_X
#----------------------------------------------------------------------

if __name__ == "__main__":
    positionlist,neighlists,distlists,atomwisedistlist, exactlist = pickle.load( open( DATA_LOCATION, "rb" ) )
    test_positionlist,test_neighlists,test_distlists,test_atomwisedistlist,test_exactlist = pickle.load( open( TEST_DATA_LOCATION, "rb" ) )

    print('Length of Train set: %s'%len(exactlist))
    print('Length of Test set: %s'%len(test_exactlist))
    sys.stdout.flush()


    X, X_test = prepare_input_data(positionlist, distlists, atomwisedistlist,
                           test_positionlist, test_distlists, test_atomwisedistlist)


    model = ForcefieldModel(rcut=RCUT, symbasis=activation_funcs)
#model.load_model(modelname)


    print("####### Starting MC Run %s ########"%mc_run)

# Initialize Random weights
    curweight = model.get_npweights()
    cnt = -1
    for i, row in enumerate(curweight):
        for j, x in np.ndenumerate(row):
            cnt += 1
            #curweight[i][j] = mask[cnt]
            curweight[i][j] = random.uniform(-1, 1)
    model.set_npweights(curweight)
    print("Number of Parameters in the Model: %s"%(cnt+1))


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    current_loss = 10000

    for epoch in range(n_epochs):
        try:
            vars, train_loss, grads = grad(model,X,exactlist) # Train Error and gradients
            test_loss = loss(model, X_test, test_exactlist)   # Test Error
            print('Epoch:',epoch, 'Train LOSS:', train_loss.numpy(), 'Test LOSS:', test_loss.numpy())

            if print_exprs:
                _ = model.pretty_output(threshold=0.01)

            if (train_loss < current_loss):
                current_loss = train_loss

                model.save_model(SAVEDIR + 'symmodel_epoch%s.ml'%epoch)
                model.save_model(SAVEDIR + 'symmodel_best.ml')

            optimizer.apply_gradients(zip(grads, vars))
            sys.stdout.flush()
        except:
            print('Error encountered, Moving to Next MC Run')
            break
