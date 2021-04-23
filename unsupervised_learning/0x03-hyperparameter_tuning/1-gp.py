#!/usr/bin/env python3
""" Gaussian Process Prediction """

import numpy as np


class GaussianProcess:
    """ Class that represents a noiseless 1D Gaussian process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Init constructor """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Method that calculates the covariance kernel matrix
            between two matrices """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """ Method that predicts the mean and standard deviation
            of points in a Gaussian process """
        covarianceKernelMatrix = self.kernel(self.X, X_s)
        secondCKM = self.kernel(X_s, X_s)
        inverseK = np.linalg.inv(self.K)
        mu = covarianceKernelMatrix.T.dot(inverseK).dot(self.Y)
        mu = mu.reshape(-1)
        product = covarianceKernelMatrix.T.dot(inverseK)
        sigma = np.diag(secondCKM - product.dot(covarianceKernelMatrix))

        return mu, sigma
