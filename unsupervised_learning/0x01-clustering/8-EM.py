#!/usr/bin/env python3
""" EM """

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ Function that performs the expectation maximization for a GMM """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None, None)
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return (None, None, None, None, None)
    if not isinstance(iterations, int) or iterations <= 0:
        return (None, None, None, None, None)
    if not isinstance(tol, float) or tol < 0:
        return (None, None, None, None, None)
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    actualLK = 0

    for i in range(iterations):
        g, lk = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if verbose:
            message = 'Log Likelihood after {} iterations: {}'\
                .format(i, lk.round(5))

            if (i % 10 == 0) or (i == 0):
                print(message)

            if abs(lk - actualLK) <= tol:
                print(message)
                break

        if abs(lk - actualLK) <= tol:
            break

        actualLK = lk

    return pi, m, S, g, lk
