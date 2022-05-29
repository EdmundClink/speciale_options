#Building the ML model
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU, ELU, ReLU
from keras import backend

def loadData(path_to_file):
    # Import dataset
    data = pd.read_csv(path_to_file, sep=',')

    # Remove NaN values due to the rolling volatility
    data = data.dropna()

    return data

def splitTrainAndTest(X, Y, test_size):  
    data_split_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    return data_split_test

def custom_activation(x):
    return backend.exp(x)

#NN MODEL 1
def build_BS(X):
    nodes = 120
    model = Sequential()

    model.add(Dense(nodes, input_dim=X.shape[1]))
    model.add(ReLU())

    model.add(Dense(1)) #output layer
    model.add(Activation(custom_activation))

    model.compile(loss='mse',optimizer='rmsprop')

    model.summary()
    return model


#NN MODEL 2
def build(X):
    nodes = 120
    model = Sequential()

    model.add(Dense(nodes, input_dim=X.shape[1]))
    model.add(LeakyReLU())
    #model.add(Dropout(0.1))

    model.add(Dense(nodes, activation='elu'))
    #model.add(Dropout(0.1))

    model.add(Dense(nodes, activation='relu'))
    #model.add(Dropout(0.1))

    model.add(Dense(nodes, activation='elu'))
    #model.add(Dropout(0.1))

    model.add(Dense(1))
    model.add(Activation(custom_activation))

    model.compile(loss='mse',optimizer='rmsprop')

    model.summary()
    return model