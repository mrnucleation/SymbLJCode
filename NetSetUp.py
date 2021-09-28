import sys
#sys.path.append("/home/bishop/Kim_SymbNeuralNet/")
import functions
from symbolic_network import SymbolicNet, MaskedSymbolicNet, SymbolicNetL0
from inspect import signature
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

def getmodel(nlayer=2, inputdim=1, actfunc=None, masked_net=False, l0_loss=False):
    init_sd_first = 0.1
    init_sd_last = 1.0
    init_sd_middle = 0.5
    if actfunc is None:
        activation_funcs = [
            *[functions.Constant()] * 1,
            *[functions.Identity()] * 1,
            *[functions.Square()] * 1,
            *[functions.Sin()] * 1,
            *[functions.Cos()] * 1,
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2,
        ]
    else:
        activation_funcs = actfunc
    
    symnn = MaskedSymbolicNet if masked_net else SymbolicNet
    symnn = SymbolicNetL0 if l0_loss else SymbolicNet
        
    #inputdim = 1  # Number of input arguments to the function
    
    if all(isinstance(elem, list) for elem in activation_funcs):
        # Seperate act function for each layer
        nlayer = len(activation_funcs)
        width = len(activation_funcs[0])
        n_double = functions.count_double(activation_funcs[0])
        initialweights = [tf.zeros([inputdim, width + n_double])]        
        for i in range(1, nlayer):
            inputdim = width
            width = len(activation_funcs[i])
            n_double = functions.count_double(activation_funcs[i])
            initialweights = initialweights + [tf.zeros([inputdim, width + n_double])]
        initialweights = initialweights + [tf.zeros([width, 1])]
        
    else:
        width = len(activation_funcs)
        n_double = functions.count_double(activation_funcs)
        initialweights = [tf.zeros([inputdim, width + n_double])] 
        initialweights = initialweights + [tf.zeros([width, width + n_double])  for i in range(1, nlayer)] 
        initialweights = initialweights + [tf.zeros([width, 1])]
    
    sym = symnn(nlayer,
            funcs=activation_funcs,
            initial_weights=initialweights
    )
    return sym

def lossfunc(Y_predict, Y_target):
    err = tf.math.subtract(Y_predict, Y_target)
    err = tf.math.abs(err)
    tol = 1e1
    cutoff = 5e-2
    slope = (tol-0.0)/(1.0-cutoff)
    intercept = tol - slope*1.0

    pen = tf.math.divide(err, tf.abs(Y_target))
    zeros = tf.zeros(shape=tf.shape(pen))
    pen = tf.map_fn(lambda x:slope*x+intercept, pen)
    pen = tf.where(pen > 5e-2, pen, zeros)
    err = tf.add(pen,err)

    score = tf.reduce_mean(err)
    return score


if __name__ == "__main__":
    model = getmodel()
