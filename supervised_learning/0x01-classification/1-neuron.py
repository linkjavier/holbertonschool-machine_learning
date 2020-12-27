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
