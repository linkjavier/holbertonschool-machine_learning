#!/usr/bin/env python3
""" Regular Chains """

import numpy as np


def regular(P):
    """ Function def regular(P): that determines the
        steady state probabilities of a regular markov chain
    """

    if len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if P.shape[0] < 1:
        return None

    MatrixPower = np.linalg.matrix_power(P, 100)

    if np.any(MatrixPower <= 0):
        return None

    SSProb = np.array([MatrixPower[0]])

    return SSProb
