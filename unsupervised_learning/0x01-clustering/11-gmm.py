#!/usr/bin/env python3
""" GMM Gaussian Mixture """

import sklearn.mixture


def gmm(X, k):
    """ Function  that calculates a GMM from a dataset """

    Gaussian = sklearn.mixture.GaussianMixture(n_components=k)
    params = Gaussian.fit(X)
    clss = Gaussian.predict(X)
    pi = params.weights_
    m = params.means_
    S = params.covariances_
    bic = Gaussian.bic(X)

    return pi, m, S, clss, bic
