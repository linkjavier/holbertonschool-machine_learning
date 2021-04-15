#!/usr/bin/env python3
""" Initialize Bayesian Optimization """

import numpy as np
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
