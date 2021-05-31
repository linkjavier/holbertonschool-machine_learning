#!/usr/bin/env python3
""" RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ Function that performs forward propagation for a simple RNN """

    t, _, _ = X.shape
    H = [h_0]
    Y = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[-1], X[step])
        H.append(h_next)
        Y.append(y)

    H = np.array(H)
    Y = np.array(Y)

    return H, Y
