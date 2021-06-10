#!/usr/bin/env python3
""" Maximization """

import numpy as np


def maximization(X, g):
    """ Function that calculates the maximization step
        in the EM algorithm for a GMM
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    gSize = g.shape[0]
    x1, x2 = X.shape

    if not np.isclose(np.sum(g, axis=0), np.ones((x1,))).all():
        return None, None, None

    pi = np.zeros((gSize,))
    m = np.zeros((gSize, x2))
    S = np.zeros((gSize, x2, x2))

    for i in range(gSize):
        m[i] = np.sum(g[i, :, None] * X, axis=0) / np.sum(g[i], axis=0)
        auxiliar = X - m[i]
        S[i] = np.dot(g[i] * auxiliar.T, auxiliar) / np.sum(g[i])
        pi[i] = np.sum(g[i]) / x1

    return pi, m, S
