#!/usr/bin/env python3
""" P affinities """

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """ Function that calculates the symmetric P affinities of a data set """

    n, _ = X.shape

    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        copyOnes = np.ones(D[i].shape, dtype=bool)
        copyOnes[i] = 0
        Hi, P[i][copyOnes] = HP(D[i][copyOnes], betas[i])
        max = None
        min = 0

        while abs(Hi - H) > tol:
            if Hi < H:
                max = betas[i, 0]
                betas[i, 0] = (max + min) / 2
            else:
                min = betas[i, 0]
                if max is None:
                    betas[i, 0] *= 2
                else:
                    betas[i, 0] = (max + min) / 2

            Hi, P[i][copyOnes] = HP(D[i][copyOnes], betas[i])

    P = (P + P.T) / (2 * n)

    return P
