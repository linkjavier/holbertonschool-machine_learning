#!/usr/bin/env python3
""" Gradients """

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ Function that calculates the gradients of Y """

    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    formula = ((P - Q) * num)
    dY = np.zeros((n, ndim))

    for i in range(n):
        aux = np.tile(formula[:, i].reshape(-1, 1), ndim)
        dY[i] = (aux * (Y[i] - Y)).sum(axis=0)

    return dY, Q
