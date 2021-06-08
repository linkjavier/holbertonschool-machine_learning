#!/usr/bin/env python3
""" PDF """

import numpy as np


def pdf(X, m, S):
    """ Function that calculates the probability denominatorsity
        function of a Gaussian distribution
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != S.shape[1] or S.shape[0] != S.shape[1]:
        return None
    if X.shape[1] != m.shape[0]:
        return None

    _, d = X.shape

    determinant = np.linalg.det(S)

    first = np.matmul((X - m), np.linalg.inv(S))
    second = np.sum(first * (X - m), axis=1)
    numerator = np.exp(second / -2)
    denominator = np.sqrt(determinant) * ((2 * np.pi) ** (d/2))
    pdf = numerator / denominator

    pdf = np.where(pdf < 1e-300, 1e-300, pdf)

    return pdf
