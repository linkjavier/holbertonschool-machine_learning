#!/usr/bin/env python3
"""  Variance """

import numpy as np


def variance(X, C):
    """ Function that calculates the total
        intra-cluster variance for a data set
    """

    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    variances = np.sum((X - C[:, np.newaxis])**2, axis=-1)
    mean = np.sqrt(variances)
    min = np.min(mean, axis=0)
    var = np.sum(min ** 2)
    answer = np.sum(var)

    return answer
