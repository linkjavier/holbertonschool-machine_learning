#!/usr/bin/env python3
""" Bayesian Optimization """

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ Class that performs Bayesian optimization
        on a noiseless 1D Gaussian process"""

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """ Init Constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.minimize = minimize
        self.xsi = xsi
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """ Method that calculates the next best sample location """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide='warn'):
            if self.minimize:
                Y_s = np.min(self.gp.Y)
                imp = (Y_s - mu - self.xsi).reshape(-1, 1)
            else:
                Y_s = np.amax(self.gp.Y)
                imp = (mu - Y_s - self.xsi).reshape(-1, 1)

            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return (X_next, EI.reshape(-1))

    def optimize(self, iterations=100):
        """ Public instance method that optimizes the black-box function """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            if (X_next == self.gp.X).any():
                self.gp.X = self.gp.X[:-1]
                break
            self.gp.update(X_next, Y_next)

        if self.minimize:
            position = np.argmin(self.gp.Y)
        else:
            position = np.argmax(self.gp.Y)

        X_opt = self.gp.X[position]
        Y_opt = self.gp.Y[position]

        return X_opt, Y_opt
