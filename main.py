#Implementation of the classic Connect4 using an ANN. Reference used is "Artificial Inteligence: A Guide to Intelligent Systems"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'Lib/site-packages/')
import numpy as np
import json

with open('option.txt', 'r') as file:
    option = json.load(file)


class NeuralNetwork:
    features = 42
    WIDTH = 7
    row = 6
    neurons = 60
    Epoch = 0
    iteration = 0
    learnRate = .1
    __init__(self, options, data):
        if (option['mode'] = 'train'):
            self.X = np.array(option['input']) #Input
            self.Yd = np.array(option['output'])#Desired output
            self.Y_InOutErr = np.zeros((3,7))#Output neurons
            self.h_lay= 1 #Number of hidden layers
            self.n_InOutErr = np.zeros((3,neurons))#hidden layer neurons
            self.err_out = np.zeros(option['output'])#error between output and desired output
            
            self.W_ih =  np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, features)) #Weights between input and hidden layer.
            self.T_h = np.random.normal(0.0, 1.0/np.sqrt(features), (neurons)) #Hidden layer neurons' tresholds 
            self.W_ho = np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, WIDTH))

    def NeuronsActivation(self, tresholds, size, neurons, inp, weights):
        #ref: TB p177 eq:6.96
        tresholds = np.reshape(tresholds, (size, 1)) #Turning from a 1D array into a matrix
        neurons = np.sum(np.subtract(np.multiply(weights, inp), tresholds), axis=1) #Calculation of the input to each neuron first
        neurons = np.tanh(neurons)  #Activation function is tanh which is a rescaling of the 
                                    #Sigmoid over -1 to 1.
    
    def outErr(self, cmd=None):
        self.err_out = np.subtract(self.Y, self.Yd)

    def backPropagation(self, inpNeur, weights, outNeur):
        #ref p177
        errGradient = np.zeros(np.shape(weights)[1])
        if (cmd)
        errGradient = np.subtract(1, np.multiply(outNeur[1], outNeur[1])*outNeur[2])
        inpNeur[1] = np.reshape(inpNeur[1], (7,1))
        weightCorr = np.multiply(learnrate, np.multiply(inpNeur[1], errGradient))
        weights = np.add(weights, weightCorr)

    def backPropagationMid(self, weights, inpNeur, outNeur):
        pass