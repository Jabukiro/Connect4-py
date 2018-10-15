#Implementation of the classic Connect4 using an ANN
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
    col = 7
    row = 6
    neurons = 60
    Epoch = 0
    iteration = 0
    learnRate = .1
    __init__(self, options, data):
        if option['mode'] = 'train':
                self.X = np.array(option['input']) #Input
                self.curr_X = self.X[iteration]
                self.Yd = np.array(option['output'])#Desired output
                self.Y = np.zeros((7))#ANN Output
                self.h_lay= 1 #Number of hidden layers
                self.n_output = np.zeros((neurons))
                self.err_out = #error between output and desired output
                
                self.W_ih =  np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, features)) #Weights between input and hidden layer.
                self.T_h = np.random.normal(0.0, 1.0/np.sqrt(features), (neurons)) #Hidden layer neurons' tresholds 
                        
        def ih_act(self):
            #ref: TB p177 eq:6.9
            self.T_h = np.reshape(self.T_h, (neurons, 1)) #Turning from a 1D array into a matrix
            self.neuron = np.sum(np.subtract(np.multiply(self.W_ih, self.curr_X), self.T_h), axis=1) #Calculation of the input to each neuron first
            self.neuron = np.tanh(self.neuron)  #Activation function is tanh which is a rescaling of the 
                                                #Sigmoid over -1 to 1.
        def action(self):
            pass