#!/usr/bin/env python3
"""Normalization Module"""
import numpy as np


def normalization_constants(X):
    """ Function that calculates the normalization
        (standardization) constants of a matrix
    """

    normalizeX = X.copy()
    m, nx = normalizeX.shape
    mean = (1 / m) * normalizeX.sum(axis=0)

    normalizeX -= mean
    normalizeX = normalizeX ** 2

    variance = (1 / m) * normalizeX.sum(axis=0)
    standardDeviation = np.sqrt(variance)

    return mean, standardDeviation
