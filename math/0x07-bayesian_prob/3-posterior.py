#!/usr/bin/env python3
""" Posterior Function """

import numpy as np


def posterior(x, n, P, Pr):
    """ FUnction that calculates the posterior probability for the various
        hypothetical probabilities of developing severe side effects given
        the data
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        m = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(m)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        m = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(m)
    for i in range(len(P)):
        if not (P[i] >= 0 and P[i] <= 1):
            a = 'All values in P must be in the range [0, 1]'
            raise ValueError(a)
        if not (Pr[i] >= 0 and Pr[i] <= 1):
            a = 'All values in Pr must be in the range [0, 1]'
            raise ValueError(a)
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    nFactorial = np.math.factorial(n)
    xFactorial = np.math.factorial(x)
    differenceFactorial = np.math.factorial(n - x)

    combination = nFactorial / (xFactorial * differenceFactorial)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))
    intersection = Pr * likelihood
    marginalProbability = np.sum(intersection)
    posterior = intersection / marginalProbability

    return posterior
