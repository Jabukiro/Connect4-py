#Implementation of the classic Connect4 using an ANN. Reference used is "Artificial Inteligence: A Guide to Intelligent Systems"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'Lib/site-packages/')
import numpy as np
import json
#TODO: Add error plotting.
#TODO: Implement momentum.
#TODO: Implement play option.
#TODO: Consider calling update() inside backpropagation(). Check error free first.

class NeuralNetwork:
    features = 42
    OUT = 1
    row = 6
    neurons = 60
    Epoch = 0
    MaxEpoch = 500
    iteration = 0
    learnRate = .1
    __init__(self, option, data):
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
        return

    def NeuronsActivation(self, container[], lidx=0, widx=0, tidx=0):
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
        return

    def backPropagation(self, err, container=[], lidx=self.totalLayers-1, widx=self.totalLayers-2, tidx=self.totalLayers):
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

    def saveState(self):
        weights = []
        tresholds = []
        for i in range(self.totalLayers-1):
            weights = self.weiList[i]
            tresholds = self.treList[i]

        options['weights'] = weights
        options['tresholds'] = tresholds
        with open('option.json', 'w') as file:
            json.dump(option, file)
        return


def main():
    with open('option.json', 'r') as file:
        option = json.load(file)
    with open('data.txt', 'r') as file:
        content = file.read()
        data = c.split().split(',')
    ann = NeuralNetwork(option, data)#Step1&2: Initialisation
    while (NeuralNetwork.Epoch <= NeuralNetwork.MaxEpoch):
        while(ann.iteration <= ann.Games_Num):
            #Activation of weights and thresholds
            ann.NeuronsActivation()
            #Weight training
            error = ann.outErr()
            ann.backPropagation(error)
            NeuralNetwork.iteration +=1
        NeuralNetwork.Epoch +=1
    ann.saveState()
    
if __name__ == '__main__':
    #Main will not run if NeuralNetwork is imported as a module
    #Can they be used to play by loading a saved state
    main()