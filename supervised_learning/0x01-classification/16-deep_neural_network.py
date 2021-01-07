#!/usr/bin/env python3
""" Python program to implement a single neuron neural network """
import numpy as np


class DeepNeuralNetwork():
    """
        Class that defines a neural network with
        one hidden layer performing binary classification.

        W1: The weights vector for the hidden layer. Upon instantiation,
            it should be initialized using a random normal distribution.
        b1: The bias for the hidden layer. Upon instantiation,
            it should be initialized with 0’s.
        A1: The activated output for the hidden layer. Upon instantiation,
            it should be initialized to 0.
        W2: The weights vector for the output neuron. Upon instantiation,
            it should be initialized using a random normal distribution.
        b2: The bias for the output neuron. Upon instantiation,
            it should be initialized to 0.
        A2: The activated output for the output neuron (prediction).
            Upon instantiation, it should be initialized to 0.
    """

    def __init__(self, nx, layers):

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # # He et algorithm

        for lay in range(self.L):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if lay == 0:
                # He et al. initialization for weights in first layer
                He = (np.random.randn(layers[lay], nx)
                      * np.sqrt(2 / nx))
                self.weights["W{}".format(lay + 1)] = He
            else:
                # He et al. initialization for weights
                He = (np.random.randn(layers[lay], layers[lay - 1])
                      * np.sqrt(2 / layers[lay - 1]))
                self.weights["W{}".format(lay + 1)] = He
            # Zero initialization for biases
            self.weights["b{}".format(lay + 1)] = np.zeros((layers[lay], 1))
