#!/usr/bin/env python3
""" The Forward Algorithm """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """  Function that performs the forward
        algorithm for a hidden markov model
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

    T = Observation.shape[0]
    N, _ = Emission.shape
    F = np.zeros([N, T], dtype='float')
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    for i in range(1, T):
        F[:, i] = np.multiply(Emission[:, Observation[i]],
                              np.dot(Transition.T, F[:, i - 1]))

    return (np.sum(F[:, T - 1]), F)
