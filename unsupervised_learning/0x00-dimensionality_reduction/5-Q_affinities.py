#!/usr/bin/env python3
""" Q affinities """
import numpy as np


def Q_affinities(Y):
    """  Function that calculates the Q affinities """

    x = np.sum(Y ** 2, axis=1)
    y = np.sum(Y ** 2, axis=1)[:, np.newaxis]
    z = np.matmul(Y, Y.T)
    D = x - 2 * z + y

    numerator = (1 + D) ** (-1)
    np.fill_diagonal(numerator, 0.)
    denominator = np.sum(numerator)
    Q = numerator / denominator

    return Q, numerator
