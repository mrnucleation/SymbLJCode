import numpy as np
import sys, os
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as model_selection
from math import isnan, fabs
from AnModel import model
import tensorflow as tf

try: filename = sys.argv[1]
except IndexError: filename = "dumpfile.dat"
#del os.environ['CUDA_VISIBLE_DEVICES']
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


X = []
Y = []
with open(filename, "r") as infile:
    for i, line in enumerate(infile):
        col = line.split("|")
        eng_total = float(col[-1])
        if isnan(eng_total) or eng_total > 1e4:
            continue
        eng_train = float(col[-2])
        eng_test = float(col[-3])
        coords = [float(x) for x in col[0].split()]
        X.append(coords)
        Y.append(eng_total)

X = np.array(X)
Y = np.divide(np.array(Y), 1e4)
print(X.shape)
print(Y.shape)

npar = X.shape[1]
anmodel = model(npar)
print(npar)

#for i in range(npar):
#    reg = LinearRegression()
#    reg.fit(X[:,i].reshape(-1,1) ,Y)
#    score = reg.score(X[:,i].reshape(-1,1),Y)
#    reg.fit(X ,Y)
#    score = reg.score(X, Y)
#    print(i,score)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

if os.path.exists("model.analyze"):
#    anmodel.loadweights()
    print("Model Loaded!")
anmodel.train(X_train, Y_train, nepoch=15000)
anmodel.saveweights()


with tf.GradientTape() as tape:
    results = anmodel(X_train)
print(results)
print(results.shape)
testresults = anmodel(X_test)

with open("an_corr_train.dat", "w") as outfile:
    for y_train, y_target in zip(Y_train, results):
        outfile.write("%s %s\n"%(y_train*1e4, y_target[0]*1e4))
with open("an_corr_test.dat", "w") as outfile:
    for y_test, y_target in zip(Y_test, results):
        outfile.write("%s %s\n"%(y_test*1e4, y_target[0]*1e4))
