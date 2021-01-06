#!/usr/bin/env python3
""" Python program to implement a single neuron neural network """
import numpy as np


class NeuralNetwork():
    """
        Class that defines a neural network with
        one hidden layer performing binary classification.
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
