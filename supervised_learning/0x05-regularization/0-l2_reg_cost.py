#!/usr/bin/env python3
""" L2 Regularization Cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Function that calculates the cost of a
        neural network with L2 regularization
    """

    AddReg = 0
    for i in range(1, L + 1):
        AddReg += np.linalg.norm(weights.get('W' + str(i)))
    return cost + AddReg * lambtha / (2 * m)
