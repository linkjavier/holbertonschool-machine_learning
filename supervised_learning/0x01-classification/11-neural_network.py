#!/usr/bin/env python3
""" Python program to implement a single neuron neural network """
import numpy as np


class NeuralNetwork():
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

    def __init__(self, nx, nodes):
        """ Init method for Neuron Class """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """" Method to get private W1 """
        return self.__W1

    @property
    def b1(self):
        """" Method to get private b1 """
        return self.__b1

    @property
    def A1(self):
        """" Method to get private A1 """
        return self.__A1

    @property
    def W2(self):
        """" Method to get private W2 """
        return self.__W2

    @property
    def b2(self):
        """" Method to get private b2 """
        return self.__b2

    @property
    def A2(self):
        """" Method to get private A2 """
        return self.__A2

    def forward_prop(self, X):
        """
            Add the public method def forward_prop with
            sigmoid activation function. Matrix Mult. of Weights
            and X values (input data).
        """

        HiddenLayerValue = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-HiddenLayerValue))

        OutputProducedValue = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-OutputProducedValue))

        return self.__A1, self.A2

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
