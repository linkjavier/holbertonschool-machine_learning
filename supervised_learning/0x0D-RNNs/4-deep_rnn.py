#!/usr/bin/env python3
""" Deep RNN """

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ Function that performs forward propagation for a deep RNN """

    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    counter = 0

    for i in range(t):
        x1 = X[i]
        ht = np.zeros((l, m, h))
        for j in range(l):
            h_next = H[counter][j]
            h_next, pt = rnn_cells[j].forward(h_next, x1)
            x1 = h_next
            ht[j] = h_next

        Y[i] = pt
        H[i + 1] = ht
        counter += 1

    return H, Y
