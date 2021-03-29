#!/usr/bin/env python3
""" Bayesian Probability """

import numpy as np


def likelihood(x, n, P):
    """
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
        probabilitie of developing severe side effects
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if (not isinstance(P, np.ndarray)) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihood = ((np.math.factorial(n) / (np.math.factorial(x) *
                  np.math.factorial(n - x)))) * (P ** x) * ((1 - P) ** (n - x))

    return likelihood
