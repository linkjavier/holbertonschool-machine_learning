#!/usr/bin/env python3
"""Policy and policy_gradient methods"""

import numpy as np


def policy(matrix, weight):
    """ Function that computes to policy with a weight of a matrix """

    matrixDot = np.dot(matrix, weight)
    exponent = np.exp(matrixDot)
    policy = exponent / np.sum(exponent)

    return (policy)


def policy_gradient(state, weight):
    """ Function that computes the Monte-Carlo policy gradient
        based on a state and a weight matrix
    """

    policyComputed = policy(state, weight)
    action = np.random.choice(len(policyComputed[0]), p=policyComputed[0])
    s = policyComputed.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    dsoftmax = softmax[action, :]
    dlog = dsoftmax / policyComputed[0, action]
    gradient = state.T.dot(dlog[None, :])

    return action, gradient
