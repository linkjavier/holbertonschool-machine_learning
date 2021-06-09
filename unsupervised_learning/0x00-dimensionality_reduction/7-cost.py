#!/usr/bin/env python3
""" Cost """
import numpy as np


def cost(P, Q):
    """ Function that calculates the cost of the t-SNE transformation """

    toArray = np.array([[1e-12]])
    logOperator = P / np.maximum(Q, toArray)
    log = np.log(np.maximum(logOperator, toArray))
    product = (P * log)
    C = np.sum(product)

    return C
