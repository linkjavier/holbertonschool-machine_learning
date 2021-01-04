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

    def evaluate(self, X, Y):
        """
            Method to Evaluates the neuron’s predictions.
            X : is a numpy.ndarray with shape (nx, m)
                that contains the input data.
            Y : is a numpy.ndarray with shape (1, m)
                that contains the correct labels for the input data.
        """

        self.forward_prop(X)
        NeuronEvaluation = np.where(self.A <= 0.5, 0, 1), self.cost(Y, self.A)

        return NeuronEvaluation

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
            X : a numpy.ndarray with shape (nx, m) that contains the input data
            nx : the number of input features to the neuron
            m : the number of examples
            Y : a numpy.ndarray with shape (1, m) that contains,
                the correct labels for the input data
            A : a numpy.ndarray with shape (1, m) containing the
                activated output of the neuron for each example
            alpha : is the learning rate
            Updates the private attributes __W and __b
        """

        NumberOfElements = Y.shape[1]
        Difference = A - Y
        Gradient = np.matmul(Difference, X.T) / NumberOfElements
        db = np.sum(Difference) / NumberOfElements

        self.__W -= Gradient * alpha
        self.__b -= db * alpha
