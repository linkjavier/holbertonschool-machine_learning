#!/usr/bin/env python3
""" The Backward Algorithm """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ Function that performs the backward algorithm
        for a hidden markov model
    """

    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return (None, None)
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(
            Transition.shape) != 2 or \
            Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0] != Transition.shape[0] !=\
       Initial.shape[0]:
        return None, None
    if Initial.shape[1] != 1:
        return None, None

    alphaShape = Observation.shape[0]
    betaShape, _ = Emission.shape
    B = np.empty([betaShape, alphaShape], dtype='float')
    B[:, alphaShape - 1] = 1

    for t in reversed(range(alphaShape - 1)):
        B[:, t] = np.dot(Transition, np.multiply(
            Emission[:, Observation[t + 1]], B[:, t + 1]))

    P = np.dot(Initial.T, np.multiply(Emission[:, Observation[0]], B[:, 0]))

    return (P, B)
