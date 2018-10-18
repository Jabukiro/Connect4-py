#Implementation of the classic Connect4 using an ANN. Reference used is "Artificial Inteligence: A Guide to Intelligent Systems"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'Lib/site-packages/')
import numpy as np
import json
#TODO: Initialise input in neuron activation

class NeuralNetwork:
    features = 42
    WIDTH = 7
    row = 6
    neurons = 60
    Epoch = 0
    MaxEpoch = 0
    iteration = 0
    learnRate = .1
    __init__(self, options, data):
        if (option['mode'] = 'train'):
            self.Games_Num = option['GameSets']
            self.X = np.array(option['input']) #Input
            self.Yd = np.array(option['output'])#Desired output
            self.Y_InOutErr = np.zeros((3,7))#Output neurons
            self.h_lay= 1 #Number of hidden layers
            self.h1_InOutErr = np.zeros((3,neurons))#hidden layer neurons
            self.err_out = np.zeros(option['output'])#error between output and desired output
            
            self.W_ih =  np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, features)) #Weights between input and hidden layer.
            self.T_h = np.random.normal(0.0, 1.0/np.sqrt(features), neurons) #Hidden layer neurons' tresholds 
            self.W_ho = np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, WIDTH))
            self.T_o = np.random.normal(0.0, 1.0/np.sqrt(features), WIDTH)
            L_LIST = (self.X, self.h1_InOutErr, self.Y_InOutErr)
            W_LIST = (self.W_ih, self.W_ho)
            T_LIST = (self.T_h, self.T_o)

    def NeuronsActivation(self, container, lidx, widx, tidx):
        #ref: TB p177 eq:6.96
        if lidx = 0:
           container.append()
           self.NeuronsActivation(container, lidx+1, widx, tidx)
           
        elif (lidx <=2):
            inp, weights, tresholds = L_LIST[lidx], W_LIST[widx], T_LIST[tidx]

            #Calculation of the input to each neuron first
            tresholds = np.reshape(tresholds, (np.shape(tresholds)[0], 1)) #Turning from a 1D array into a matrix
            neurons[0] = np.sum(np.subtract(np.multiply(weights, inp), tresholds), axis=1)

            #Activation function is tanh which is a rescaling of the Sigmoid over -1 to 1.
            neurons[1] = np.tanh(neurons[0])

            container.append(neurons[1])
            self.NeuronsActivation(container, lidx+1, widx+1, tidx+1)
        return container
            
    
    def outErr(self, cmd=None):
        self.Y_InOutErr[2] = np.subtract(self.Y[1], self.Yd)

    def backPropagation(self, inpNeur, weights, outNeur):
        """ Updates weights between hiddenb and output layer
        @inpNeur: input
        """
        #TODO: compute the error for the next error gradient calculation, so as to implement the self calling method easily.
        #ref p177
        errGradient = np.zeros(np.shape(weights)[1])
        errGradient = np.subtract(1, np.multiply(outNeur[1], outNeur[1])*outNeur[2])
        inpNeurT = np.reshape(inpNeur[1], (7,1))
        weightCorr = np.multiply(learnrate, np.multiply(inpNeurT, errGradient))
        weights = np.add(weights, weightCorr)

    def backPropagationMid(self, inpNeur = self.X, weights =self.W_ih, outNeur = self.n_InOutErr, weights2 = self.W_ho, outNeur2 = self.Y_InOutErr):
        outNeur[2] = 1-np.multiply(np.substract(1, np.multipy(outNeur[1], outNeur[1])), np.sum(np.multiply(outNeur2[2], weights2), axis=0))
        inpNeurT = np.reshape(inpNeur[1], (60,1))
        weightCorr = np.multiply(learnrate, np.multiply(inpNeurT, outNeur[2]))
        weights = np.add(weights, weightCorr)

def main():
    with open('option.txt', 'r') as file:
        option = json.load(file)
    with open('data.txt', 'r') as file:
        data = json.load(file)
    ann = NeuralNetwork(option, data)#Step1&2: Initialisation and Activation
    while (NeuralNetwork.Epoch <= NeuralNetwork.MaxEpoch):
        while(ann.iteration <= ann.Games_Num):
            #Initialisation of hidden and output layer
            ann.NeuronsActivation(ann.T_h, ann.n_InOutErr, ann.X, ann.W_ih)
            ann.NeuronsActivation(ann.T_o, ann.Y_InOutErr, ann.n_InOutErr, ann.W_ho)
            #Weight training
            ann.outErr()
            ann.backPropagation(ann.h1_InOutErr, ann.W_ho, ann.Y_InOutErr)
            ann.backPropagationMid()
            NeuralNetwork.iteration +=1
        NeuralNetwork.Epoch +=1
