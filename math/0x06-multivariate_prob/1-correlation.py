#!/usr/bin/env python3
""" Correlation """

import numpy as np


def correlation(C):
    """ Function that calculates a correlation matrix """

    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    variance = np.diag(C).reshape(1, -1)
    deviation = np.sqrt(variance)
    standardMatrix = np.dot(deviation.T, deviation)
    correlationMatrix = C / standardMatrix

    return correlationMatrix
