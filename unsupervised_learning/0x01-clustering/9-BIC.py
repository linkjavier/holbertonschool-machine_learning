#!/usr/bin/env python3
""" BIC Clustering """

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ Function that finds the best number of clusters for a GMM
        using the Bayesian Information Criterion
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] < kmax:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    b = []
    auxiliar = []
    ks = []
    lk = []

    for k in range(kmin, kmax + 1):
        ks.append(k)
        pi, m, S, g, lkk = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)
        auxiliar.append((pi, m, S))
        lk.append(lkk)
        p = k - 1 + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * lkk
        b.append(bic)

    lk = np.array(lk)
    b = np.array(b)

    minimumIndex = np.argmin(b)
    best_k = ks[minimumIndex]
    best_result = auxiliar[minimumIndex]

    return best_k, best_result, lk, b
