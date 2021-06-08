#!/usr/bin/env python3
""" Initialize GMM """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ Function that initializes variables for a Gaussian Mixture Mode """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None)

    if type(k) is not int or k <= 0:
        return (None, None, None)

    _, d = X.shape
    m, _ = kmeans(X, k)
    pi = np.ones(k) / k
    identity = np.identity(d).reshape(-1)
    S = (np.tile(identity, k)).reshape(k, d, d)

    return (pi, m, S)
