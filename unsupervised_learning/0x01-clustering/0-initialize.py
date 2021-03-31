#!/usr/bin/env python3
""" Initialize K-means """

import numpy as np


def initialize(X, k):
    """ Function that initializes cluster centroids for K-means """

    if not isinstance(X, np.ndarray):
        return None
    if not isinstance(k, int):
        return None
    if len(X.shape) != 2:
        return None
    if k < 0:
        return None
    if k == 0:
        return None

    n, d = X.shape

    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)

    return np.random.uniform(min, max, size=(k, d))
