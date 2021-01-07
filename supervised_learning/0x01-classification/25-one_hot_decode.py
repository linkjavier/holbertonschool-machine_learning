#!/usr/bin/env python3
"""Python program that converts a one-hot matrix
    into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """
        one_hot : is a one-hot encoded numpy.ndarray with shape (classes, m)
        m : is the number of examples
        classes : is the maximum number of classes found in Y
        Returns: a one-hot encoding of Y with shape (classes, m),
        or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) is not 2:
        return None

    decode = np.argmax(one_hot, axis=0)

    return decode
