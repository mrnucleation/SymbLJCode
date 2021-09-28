import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

#================================================================================
class model():
    #----------------------------------------
    def __init__(self, inputdim=53):
        self.model = Sequential()
        self.model.add(Dense(60, activation='tanh', input_dim=inputdim, use_bias=True))
        self.model.add(Dense(60, activation='tanh', use_bias=True))
        self.model.add(Dense(1, activation='sigmoid', use_bias=True))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    #----------------------------------------  
    def __call__(self, X):
        results = self.model.predict(X)
        return results
    #----------------------------------------  
    def train(self, X, Y, nepoch=100):
        self.model.fit(X, Y, epochs=nepoch, batch_size=32)
    #----------------------------------------
    def saveweights(self, filename="model.analyze"):
        self.model.save(filename)

    #----------------------------------------
    def loadweights(self, filename="model.analyze"):
        self.model = load_model(filename)
    #----------------------------------------
    def getweights(self):
        return self.model.get_weights()
    #----------------------------------------
    def setweights(self, inweights):
        self.model.set_weights(inweights)
    #----------------------------------------
