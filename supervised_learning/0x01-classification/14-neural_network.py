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

    def evaluate(self, X, Y):
        """
            Method to Evaluates the neuron’s predictions.
            X : is a numpy.ndarray with shape (nx, m)
                that contains the input data.
            Y : is a numpy.ndarray with shape (1, m)
                that contains the correct labels for the input data.
        """

        self.forward_prop(X)
        NeuronEvaluation = np.where(
            self.A2 <= 0.5, 0, 1), self.cost(
            Y, self.A2)

        return NeuronEvaluation

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network
            X is a numpy.ndarray with shape (nx, m) that
            contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
            Y is a numpy.ndarray with shape (1, m) that contains
            the correct labels for the input data
            A1 is the output of the hidden layer
            A2 is the predicted output
            alpha is the learning rate
            Updates the private attributes __W1, __b1, __W2, and __b2
        """

        NumberOfElements = Y.shape[1]

        Difference = A2 - Y
        GradientW2 = np.matmul(Difference, A1.T) / NumberOfElements
        biasAverageOutput = np.sum(
            Difference, axis=1, keepdims=True) / NumberOfElements

        OutputTransformation = np.matmul(
            self.W2.T, Difference) * (A1 * (1 - A1))
        GradientW1 = np.matmul(OutputTransformation, X.T) / NumberOfElements
        hiddenbiasAverage = np.sum(
            OutputTransformation,
            axis=1,
            keepdims=True) / NumberOfElements

        self.__W1 -= GradientW1 * alpha
        self.__b1 -= hiddenbiasAverage * alpha

        self.__W2 -= GradientW2 * alpha
        self.__b2 -= biasAverageOutput * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            Trains the neuron
            X is a numpy.ndarray with shape (nx, m) that contains
            the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y is a numpy.ndarray with shape (1, m) that contains
            the correct labels for the input data
            iterations is the number of iterations to train over
                if iterations is not an integer, raise a TypeError with the
                exception iterations must be an integer
                if iterations is not positive, raise a ValueError with the
                exception iterations must be a positive integer
            alpha is the learning rate
                if alpha is not a float, raise a TypeError with the
                exception alpha must be a float
                if alpha is not positive, raise a ValueError with the
                exception alpha must be positive
            All exceptions should be raised in the order listed above
            Updates the private attributes __W, __b, and __A
            You are allowed to use one loop
            Returns the evaluation of the training data after iterations
            of training have occurred
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        return self.evaluate(X, Y)
