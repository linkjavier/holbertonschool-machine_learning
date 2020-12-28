#!/usr/bin/env python3
""" Python program to implement a single neuron neural network """
import numpy as np


class Neuron():
    """
        Class Neuron that defines a single neuron
        performing binary classification.
    """

    def __init__(self, nx):
        """ Init method for Neuron Class """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Method to get private W"""
        return self.__W

    @property
    def b(self):
        """ Method to get private b"""
        return self.__b

    @property
    def A(self):
        """ Method to get private A"""
        return self.__A

    @property
    def A(self):
        """ Method to get private A"""
        return self.__A

    def forward_prop(self, X):
        """
            Add the public method def forward_prop with
            sigmoid activation function. Matrix Mult. of Weights
            and X values (input data).
        """

        NeuronProducedValues = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-NeuronProducedValues))
        return self.__A

    def cost(self, Y, A):
        """
            Method to calculate the cost of the model
            using logistic regression
            Y : contains the correct labels for the input data.
            A : contains the activated output of the neuron for each.
        """

        cost = - (1 / Y.shape[1]) * np.sum(np.multiply(Y, np.log(A)) +
                                           np.multiply(1 - Y,
                                                       np.log(1.0000001 - A)))
        return cost
