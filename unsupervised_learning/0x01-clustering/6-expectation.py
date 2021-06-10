#!/usr/bin/env python3
""" Expectation """


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ Function that calculates the expectation step
        in the EM algorithm for a GMM
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    n, d = X.shape

    if d != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)
    if d != m.shape[1] or m.shape[0] != S.shape[0]:
        return (None, None)
    if pi.shape[0] != m.shape[0]:
        return (None, None)

    if not np.isclose(np.sum(pi), 1):
        return None, None

    k = S.shape[0]
    auxiliar = np.zeros((k, n))

    for i in range(k):
        PDF = pdf(X, m[i], S[i])
        auxiliar[i] = pi[i] * PDF

    g = auxiliar / np.sum(auxiliar, axis=0)
    tll = np.sum(np.log(np.sum(auxiliar, axis=0)))

    return g, tll
