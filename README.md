# Connect4-py

Implementation of the classic Connect4 using an Artificial Neural Network.

Structure
----------

Numpy arrays are used

Each Layer is made up of a 3D numpy array. Contains:

    * Input
    * Output
    * Error Gradient


Weights are made up of a 2d numpy array and their shape is of (RightLayerSize, LeftLayerSize)

Activation of Neurons and BackPropagation are optimised so as to not be touched if the following parameters are changed:

    -Number of Hidden Layers
    -Number of Neurons for each Hidden Layer
    -Number of weights


**Works on Riddles.io**
