#!/usr/bin/env python3
""" Initialize """

import numpy as np


class MultiNormal:
    """ Class that represents a Multivariate Normal dimessageibution """

    def __init__(self, data):
        """ Instance contructor"""

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data, axis=1).reshape((data.shape[0], 1))

        self.mean = mean
        self.cov = np.matmul(data - self.mean, data.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """ Public instance method def that calculates
            the PDF at a data point """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]

        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        dimensions = x.shape[0]

        mean = self.mean
        cov = self.cov

        covarianceDeterminant = np.linalg.det(cov)
        covarianceInverse = np.linalg.inv(cov)

        denominator = np.sqrt(((2 * np.pi) ** dimensions)
                              * covarianceDeterminant)
        exponent = -0.5 * \
            np.matmul(np.matmul((x - mean).T, covarianceInverse), x - mean)

        pdf = (1 / denominator) * np.exp(exponent[0][0])

        return pdf
