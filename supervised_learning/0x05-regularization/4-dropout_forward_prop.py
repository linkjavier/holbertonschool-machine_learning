#!/usr/bin/env python3
""" Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Function that conducts forward propagation using Dropout """

    output = dict()
    output.update({'A0': X})
    for i in range(L):
        A = output.get('A' + str(i))
        b = weights.get('b' + str(i + 1))
        w = weights.get('W' + str(i + 1))
        Z = np.matmul(w, A) + b

        if i + 1 == L:
            t = np.exp(Z)
            a = t / np.sum(t, axis=0, keepdims=True)
        else:
            dropout = np.random.rand(Z.shape[0], Z.shape[1])
            dropout = np.where(dropout < keep_prob, 1, 0)
            output.update({'D' + str(i + 1): dropout})
            a = np.tanh(Z) * dropout / keep_prob

        output.update({'A' + str(i + 1): a})

    return output
