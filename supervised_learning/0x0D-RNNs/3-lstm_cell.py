#!/usr/bin/env python3
""" LSTM Cell """

import numpy as np


class LSTMCell:
    """ LSTM unit """

    def __init__(self, i, h, o):
        """ Constructor """

        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """ Softmax activation function """

        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ Public instance method that performs
            forward propagation for one time step
        """
        input = np.concatenate((h_prev, x_t), axis=1)

        f_t = np.dot(input, self.Wf) + self.bf
        f_t = 1 / (1 + np.exp(-f_t))

        u_t = np.dot(input, self.Wu) + self.bu
        u_t = 1 / (1 + np.exp(-u_t))

        ct = np.tanh(np.dot(input, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * ct

        o_t = np.dot(input, self.Wo) + self.bo
        o_t = 1 / (1 + np.exp(-o_t))

        h_next = o_t * np.tanh(c_next)
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
