#!/usr/bin/env python3
""" Absorbing Chains """

import numpy as np


def absorbing(P):
    """ Function that determines if a markov chain is absorbing """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2 or\
       P.shape[0] != P.shape[1]:
        return None
    if np.any(P < 0):
        return None
    if np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return None

    P = P.copy()
    absorb = np.ndarray(P.shape[0])

    while True:
        mask = absorb.copy()
        absorb = np.any(P == 1, axis=0)
        if absorb.all():
            return True

        if np.all(absorb == mask):
            return False

        absorbed = np.any(P[:, absorb], axis=1)
        P[absorbed, absorbed] = 1
