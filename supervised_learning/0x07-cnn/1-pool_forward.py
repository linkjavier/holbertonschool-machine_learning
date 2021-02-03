#!/usr/bin/env python3
""" Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_strideHeightape, stride=(1, 1), mode='max'):
    """ Function that performs forward propagation over
        a pooling layer of a neural network
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kernelHeight, kernelWidth = kernel_strideHeightape
    strideHeight, strideWidth = stride

    outputH = int(((h_prev - kernelHeight) / strideHeight) + 1)
    outputW = int(((w_prev - kernelWidth) / strideWidth) + 1)

    output = np.zeros((m, outputH, outputW, c_prev))
    rangeImage = np.arange(m)

    for i_outputH in range(outputH):
        for i_outputW in range(outputW):
            s_i_outputH = i_outputH * strideHeight
            s_i_outputW = i_outputW * strideWidth
            flt = A_prev[rangeImage, s_i_outputH: kernelHeight +
                         s_i_outputH, s_i_outputW: kernelWidth + s_i_outputW]
            if mode == 'max':
                output[rangeImage, i_outputH, i_outputW] = flt.max(axis=(1, 2))
            elif mode == 'avg':
                output[rangeImage, i_outputH, i_outputW] = np.mean(
                    flt, axis=(1, 2))

    return output
