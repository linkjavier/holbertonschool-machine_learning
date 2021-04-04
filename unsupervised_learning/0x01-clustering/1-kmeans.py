#!/usr/bin/env python3
""" 0x01. Clustering """

import numpy as np


def kmeans(X, k, iterations=1000):
    """ Function that performs K-means on a dataset """

    if type(X) is not np.ndarray or type(k) is not int:
        return (None, None)
    if len(X.shape) != 2 or k < 0:
        return (None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None)
    if k == 0:
        return (None, None)

    _, d = X.shape

    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    C = np.random.uniform(min, max, size=(k, d))

    for _ in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        CentroidMeans = np.copy(C)

        for j in range(k):
            if j not in clss:
                CentroidMeans[j] = np.random.uniform(min, max)
            else:
                CentroidMeans[j] = np.mean(X[clss == j], axis=0)

# If no change in the cluster centroids occurs between iterations
# your function should return
        if (CentroidMeans == C).all():
            return (C, clss)
        else:
            C = CentroidMeans

    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)

    return (C, clss)
