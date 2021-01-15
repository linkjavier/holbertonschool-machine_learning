#!/usr/bin/env python3
"""Shuffle Module"""
import numpy as np


def shuffle_data(X, Y):
    """ Function that shuffles the data points
        in two matrices the same way
    """

    i, j = Y.shape
    Newshuffled = np.random.permutation(i)
    Xshuffled = X[Newshuffled]
    Yshuffled = Y[Newshuffled]
    return Xshuffled, Yshuffled
