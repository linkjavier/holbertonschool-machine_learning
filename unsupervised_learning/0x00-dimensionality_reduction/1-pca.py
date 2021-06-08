#!/usr/bin/env python3
""" PCA v2 """


import numpy as np


def pca(X, ndim):
    """  Function that performs PCA on a dataset """

    Xmean = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(Xmean)
    W = vh[:ndim].T
    T = np.matmul(Xmean, W)

    return T
