#!/usr/bin/env python3
"""Function that performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
        Function that performs PCA on a dataset
        
        svd - When a is a 2D array, it is factorized as u @ np.diag(s) @ vh = (u * s) @ vh,
        where u and vh are 2D unitary arrays and s is a 1D array of a's singular values.
        When a is higher-dimensional, SVD is applied in stacked mode
    """
    
    u, s, vh = np.linalg.svd(X)

    # Return the cumulative sum of the elements along a given axis.
    sum = np.cumsum(s)

    dim = []
    for i in range(len(s)):
        if ((sum[i]) / sum[-1]) >= var:
            dim.append(i)
    jx = dim[0] + 1
    return vh.T[:, :jx]
