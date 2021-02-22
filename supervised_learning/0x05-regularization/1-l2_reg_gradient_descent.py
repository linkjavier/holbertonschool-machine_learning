#!/usr/bin/env python3
"""l2 reg gradient descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Function that updates the weights and biases of a neural network
        using gradient descent with L2 regularization
    """

    layersNumber = range(L, 0, -1)
    YShape = Y.shape[1]
    dZ_prev = 0
    w = weights.copy()

    for i in layersNumber:
        A = cache['A{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        weights_i = w.get('W{}'.format(i))
        weights_n = w.get('W{}'.format(i + 1))
        biases = w.get('b' + str(i))
        if i == L:
            dZ = A - Y
        else:
            dZ = np.matmul(weights_n.T, dZ_prev) * (1 - (A * A))
        dW = np.matmul(dZ, A_prev.T) / YShape + \
            ((lambtha / YShape) * weights_i)
        db = np.sum(dZ, axis=1, keepdims=True) / YShape
        weights['W' + str(i)] = weights_i - (dW * alpha)
        weights['b' + str(i)] = biases - (db * alpha)
        dZ_prev = dZ
