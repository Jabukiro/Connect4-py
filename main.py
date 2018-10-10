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
                self.X = np.array(option['input'])
                self.curr_X = self.X[iteration]
                self.Yd = np.array(option['output'])
                self.Y = np.zeros((7))
                self.h_lay= 1
                self.n_output = np.zeros((neurons))
                self.er_out = 
                #Note the weights array is arranged from the input(features) perspective
                #Facilitates dot product.
                self.Wi =  np.random.normal(0.0, 1.0/np.sqrt(features), (features, neurons))
                self.T1 = np.random.normal(0.0, 1.0/np.sqrt(features), (neurons))
                        
        def ih_act(self):
            #ref: TB p177 eq:6.9
            
            self.neuron = np.dot(self.curr_X, self.W1) #Calculation of the input to each neuron first
            self.neuron = np.tanh(self.neuron)  #Activation function is tanh which is a rescaling of the 
                                                #Sigmoid over -1 to 1.
        def action(self):
            pass