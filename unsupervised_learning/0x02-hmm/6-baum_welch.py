#!/usr/bin/env python3
""" The Baum-Welch Algorithm """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ Function that performs backward algorithm """

    T = Observation.shape[0]
    N, _ = Emission.shape
    backward = np.empty([N, T], dtype='float')
    backward[:, T - 1] = 1

    for t in reversed(range(T - 1)):
        backward[:, t] = np.dot(Transition, np.multiply(
            Emission[:, Observation[t + 1]], backward[:, t + 1]))

    P = np.dot(Initial.T, np.multiply(
        Emission[:, Observation[0]], backward[:, 0]))

    return (backward)


def forward(Observation, Emission, Transition, Initial):
    """ Function that performs forward algorithm """

    T = Observation.shape[0]
    N, _ = Emission.shape
    forward = np.zeros([N, T], dtype='float')
    forward[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    for t in range(1, T):
        forward[:, t] = np.multiply(
            Emission[:, Observation[t]],
            np.dot(Transition.T, forward[:, t - 1]))

    return (forward)


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ Function that performs the Baum-Welch algorithm
        for a hidden markov model
    """

    if not isinstance(
            Observations,
            np.ndarray) or len(
            Observations.shape) != 1:
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

    T = Observations.shape[0]
    M, N = Emission.shape

    for _ in range(1, iterations):
        Fordward = forward(Observations, Emission, Transition, Initial)
        Backward = backward(Observations, Emission, Transition, Initial)
        zeroX = np.zeros((M, M, T - 1))

        for i in range(T - 1):
            denominator = np.dot(np.dot(
                Fordward[:, i].T, Transition) *
                    Emission[:, Observations[i + 1]].T, Backward[:, i + 1])

            for j in range(M):
                numerator = Fordward[j, i] * Transition[j] * \
                    Emission[:, Observations[i + 1]].T * Backward[:, i + 1].T
                zeroX[j, :, i] = numerator / denominator

        newMat = np.sum(zeroX, axis=1)
        Transition = np.sum(zeroX, 2) / np.sum(newMat, axis=1).reshape((-1, 1))
        newMat = np.hstack(
            (newMat, np.sum(zeroX[:, :, T - 2], axis=0).reshape((-1, 1))))
        denominator = np.sum(newMat, axis=1)

        for i in range(N):
            Emission[:, i] = np.sum(newMat[:, Observations == i], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

    return (Transition, Emission)
