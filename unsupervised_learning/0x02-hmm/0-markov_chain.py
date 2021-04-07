#!/usr/bin/env python3
""" Markov Chain """

import numpy as np


def markov_chain(P, s, t=1):
    """ Function that determines the probability of a markov chain
        being in a particular state after a specified number of iterations
    """

    if not isinstance(P, np.ndarray):
        return None
    if not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2:
        return None
    if len(s.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if s.shape[1] != P.shape[0]:
        return None
    if s.shape[0] != 1:
        return None

    EspecificProbability = np.copy(s)

    for _ in range(t):
        EspecificProbability = np.matmul(EspecificProbability, P)

    return EspecificProbability
