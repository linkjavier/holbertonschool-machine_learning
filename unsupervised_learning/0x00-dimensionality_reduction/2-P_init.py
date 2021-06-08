#!/usr/bin/env python3
""" Initialize t-SNE """
import numpy as np


def P_init(X, perplexity):
    """ Function that initializes all variables required
        to calculate the P affinities in t-SNE
    """

    n, _ = X.shape
    x = np.sum(X ** 2, axis=1)
    y = np.sum(X ** 2, axis=1)[:, np.newaxis]
    z = np.matmul(X, X.T)
    D = x - 2 * z + y
    np.fill_diagonal(D, 0.)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
