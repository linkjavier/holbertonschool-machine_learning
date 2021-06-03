#!/usr/bin/env python3
""" Bidirectional Output """

import numpy as np
from numpy.core.fromnumeric import size


class BidirectionalCell():
    """ Class that represents a bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """ Constructor """

        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Public instance method that calculates the hidden state
            in the forward direction for one time step
        """

        input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(input, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """ Public instance method that calculates the hidden state
            in the backward direction for one time step
        """

        input = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(input, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """ Public instance method that calculates all outputs for the RNN """

        t, m, _ = H.shape
        size = self.by.shape[1]
        Y = np.zeros((t, m, size))

        for i in range(t):
            y = np.dot(H[i], self.Wy) + self.by
            y = np.exp(y) / np.sum(np.exp(y), axis=1,
                                   keepdims=True)
            Y[i] = y

        return Y
