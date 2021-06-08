#!/usr/bin/env python3
""" Entropy """
import numpy as np


def HP(Di, beta):
    """ Function that calculates the Shannon entropy
        and P affinities relative to a data point
    """

    numerator = np.exp(-Di * beta)
    denominator = np.sum(numerator)
    Pi = numerator / denominator
    Hi = - np.sum(Pi * np.log2(Pi))

    return Hi, Pi
