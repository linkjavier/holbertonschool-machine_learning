#!/usr/bin/env python3
""" PCA """

import numpy as np


def pca(X, var=0.95):
    """ Function that performs PCA on a dataset """

    _, s, vh = np.linalg.svd(X)

    sum = np.cumsum(s)

    dim = []
    for i in range(len(s)):
        if ((sum[i]) / sum[-1]) >= var:
            dim.append(i)
    jx = dim[0] + 1

    return vh.T[:, :jx]
