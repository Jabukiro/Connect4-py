#Implementation of the classic Connect4 using an ANN. Reference used is "Artificial Inteligence: A Guide to Intelligent Systems"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'Lib/site-packages/')
import numpy as np
import json
#TODO: Update main()
#TODO: Consider assigning automatic values to backpropagation param so that it can be called args free
#TODO: Consider calling update() inside backpropagation(). Check error free first.

class NeuralNetwork:
    features = 42
    OUT = 1
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
            self.Y_InOutErr = np.zeros((3,OUT))#Output neurons
            self.totalLayers= option['Layers'] #Number layers
            self.h1_InOutErr = np.zeros((3,neurons))#hidden layer neurons
            self.err_out = np.zeros(option['output'])#error between output and desired output
            
            self.W_ih =  np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, features)) #Weights between input and hidden layer.
            self.T_h = np.random.normal(0.0, 1.0/np.sqrt(features), neurons) #Hidden layer neurons' tresholds 
            self.W_ho = np.random.normal(0.0, 1.0/np.sqrt(features), (neurons, OUT))
            self.T_o = np.random.normal(0.0, 1.0/np.sqrt(features), OUT)
            self.layList = [self.X, self.h1_InOutErr, self.Y_InOutErr]
            self.weiList = [self.W_ih, self.W_ho]
            self.treList = [self.T_h, self.T_o]

    def update(self, cmd, container):
        if cmd = "inputs":
            for i in range(self.totalLayers):
                self.layList[i] = container[i]
        if cmd = "weights":
            for i in range(self.totalLayers-1):
                self.weiList[i] = container[i]

    def NeuronsActivation(self, container, lidx, widx, tidx):
        #ref: TB p177 eq:6.96
        if lidx = 0:
           container.append()
           self.NeuronsActivation(container, lidx+1, widx, tidx)
           
        elif (lidx <=2):
            inp, weights, tresholds = self.layList[lidx], self.weiList[widx], self.treList[tidx]

            #Calculation of the input to each neuron first
            tresholds = np.reshape(tresholds, (np.shape(tresholds)[0], 1)) #Turning from a 1D array into a matrix
            neurons[0] = np.sum(np.subtract(np.multiply(weights, inp), tresholds), axis=1)

            #Activation function is tanh which is a rescaling of the Sigmoid over -1 to 1.
            neurons[1] = np.tanh(neurons[0])

            container.append(neurons[1])
            self.NeuronsActivation(container, lidx+1, widx+1, tidx+1)
        return container
            
    
    def outErr(self, cmd=None):
        self.Y_InOutErr[2] = np.subtract(self.Y_InOutErr[1], self.Yd)

    def backPropagation(self, container, err, lidx, widx, tidx):
        """ Recursive Function. Returns list containing new weights for next iteration, for all weights.
            Only needed to be called once per iteration.
            Should call self.update() next to actually update the
            #ref p177
        """
        #ref p177
        
        #Output Layer Error Gradient calculation
        if lidx = len(self.totalLayers)-1:
            errGradient = np.zeros(np.shape(self.layList[lidx][2])[1])
            errGradient = np.subtract(1, np.multiply(self.layList[lidx][1], self.layList[lidx][1]))*err
        #Middle layer Error Gradient Calculation
        elif lidx >0:
            errGradient = 1-np.multiply(np.substract(1, np.multipy(outNeur[1], outNeur[1])), err))

        #Rest is similar for all layers
        inpNeurT = np.reshape(self.layList[lidx-1][1], (OUT,1))
        weightCorr = np.multiply(learnrate, np.multiply(inpNeurT, errGradient))
        weights = np.add(self.weiList[widx], weightCorr)

        container.append(weights)
        self.layList[lidx][2] = errGradient

        if (widx == 0): #Means no more weights needed to be calculated
            return container

        err = np.sum(np.multiply(self.layList[lidx][2], self.weiList[widx]), axis=0
        backPropagation(container, err, lidx-1, widx-1, tidx-1)


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
