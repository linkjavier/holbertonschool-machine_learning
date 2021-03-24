#!/usr/bin/env python3
"""Function that performs PCA on a dataset"""

import numpy as np

np.random.seed(0)
a = np.random.normal(size=50)
b = np.random.normal(size=50)
c = np.random.normal(size=50)
d = 2 * a
e = -5 * b
f = 10 * c

X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]
X_m = X - np.mean(X, axis=0)

def pca(X, var=0.95):
    """
        Function that performs PCA on a dataset

        svd - When a is a 2D array, it is factorized as u @ np.diag(s) @ vh = (u * s) @ vh,
        where u and vh are 2D unitary arrays and s is a 1D array of a's singular values.
        When a is higher-dimensional, SVD is applied in stacked mode
    """
    u, s, vh = np.linalg.svd(X)

    print(s)

    # Return the cumulative sum of the elements along a given axis.
    acum = np.cumsum(s)

    dim = []
    for i in range(len(s)):
        if ((acum[i]) / acum[-1]) >= var:
            dim.append(i)
    r = dim[0] + 1

    return vh.T[:, :r]

W = pca(X_m)
# T = np.matmul(X_m, W)
# print(T)
# X_t = np.matmul(T, W.T)
# print(np.sum(np.square(X_m - X_t)) / m)

# print(np.mean(X, axis=0))