#!/usr/bin/env python3
""" PDF Multinormal """

import numpy as np


class MultiNormal():
    """ Class Multinormal """

    def __init__(self, data):
        """ Constructor Init  """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        _, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        X_mean = data - self.mean
        self.cov = np.dot(X_mean, X_mean.T) / (n - 1)

    def pdf(self, x):
        """ Method that calculates the PDF at a data point  """

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d, _ = self.cov.shape

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        determinant = np.linalg.det(self.cov)
        base = 1 / (((2 * np.pi) ** (d / 2)) * (np.sqrt(determinant)))
        factor = np.dot((x - self.mean).T, np.linalg.inv(self.cov))
        factor = np.dot(factor, (x - self.mean))
        pdf = base * np.exp((-1 / 2) * factor)
        pdf = pdf[0][0]

        return pdf
