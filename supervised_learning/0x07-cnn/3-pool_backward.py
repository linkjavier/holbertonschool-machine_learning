#!/usr/bin/env python3
"""Pool forward"""
import numpy as np


def pool_backward(
        dA,
        A_prev,
        kernel_strideHeightape,
        stride=(
            1,
            1),
        mode='max'):
    """Performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c = dA.shape
    kernelHeight, kernelWidth = kernel_strideHeightape
    strideHeight, strideWidth = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for i_oh in range(h_new):
            for i_ow in range(w_new):
                for i_nc in range(c):
                    ystrideHeight = i_oh * strideHeight
                    ystrideHeightk = ystrideHeight + kernelHeight
                    xstrideWidth = i_ow * strideWidth
                    xstrideWidthk = xstrideWidth + kernelWidth

                    if mode == 'max':
                        a = A_prev[i]
                        slice = a[ystrideHeight:ystrideHeightk,
                                  xstrideWidth:xstrideWidthk, i_nc]
                        MaxBool = (slice == np.max(slice))
                        mul = np.multiply(MaxBool, dA[i, i_oh, i_ow, i_nc])
                        dA_prev[i, ystrideHeight: ystrideHeightk,
                                xstrideWidth: xstrideWidthk, i_nc] += mul
                    elif mode == 'avg':
                        dA_var = dA[i, i_oh, i_ow, i_nc]
                        daaverage = dA_var / (kernelHeight * kernelWidth)
                        Z = np.ones(kernel_strideHeightape) * daaverage
                        dA_prev[i, ystrideHeight: ystrideHeightk,
                                xstrideWidth: xstrideWidthk, i_nc] += Z

    return dA_prev
