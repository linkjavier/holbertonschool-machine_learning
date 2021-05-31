#!/usr/bin/env python3
""" Initialize RNNCell """

import numpy as np


class RNNCell:
    """ class RNNCell that represents a cell of a simple RNN """

    def __init__(self, i, h, o):
        """ Class constructor """

        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Performs forward propagation for one time step """

        concatenate = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concatenate @ self.Wh + self.bh)
        softmax = h_next @ self.Wy + self.by
        y = np.exp(softmax) / np.sum(np.exp(softmax), axis=1, keepdims=True)

        return h_next, y
