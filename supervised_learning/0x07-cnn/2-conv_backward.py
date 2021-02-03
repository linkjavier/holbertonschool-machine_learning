#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Function that performs back propagation over
        a convolutional layer of a neural network
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kernelHeight, kernelWidth, c_prev, newChannels = W.shape
    strideHeight, strideWidth = stride

    if padding == 'valid':
        paddingHeight = 0
        paddingWidth = 0

    elif padding == 'same':
        paddingHeight = int(
            np.ceil(
                ((h_prev - 1) * strideHeight + kernelHeight - h_prev) / 2))
        paddingWidth = int(
            np.ceil(
                ((w_prev - 1) * strideWidth + kernelWidth - w_prev) / 2))

    outputH = int(
        ((h_prev + 2 * paddingHeight - kernelHeight) / strideHeight) + 1)
    outputW = int(
        ((w_prev + 2 * paddingWidth - kernelWidth) / strideWidth) + 1)

    input_pd = np.pad(
        A_prev,
        ((0, 0), (paddingHeight, paddingHeight), (paddingWidth,
                                                  paddingWidth), (0, 0)),
        'constant'
    )

    dA = np.zeros(input_pd.shape)
    dW = np.zeros(W.shape)
    db = np.sum(
        dZ,
        axis=(0, 1, 2),
        keepdims=True
    )

    for i in range(m):
        for i_outputH in range(outputH):
            for i_outputW in range(outputW):
                for i_newChannels in range(newChannels):
                    ysh = i_outputH * strideHeight
                    yshk = ysh + kernelHeight
                    xsw = i_outputW * strideWidth
                    xswk = xsw + kernelWidth
                    dZ_cut = dZ[i, i_outputH, i_outputW, i_newChannels]
                    mat_dZ_W = dZ_cut * W[:, :, :, i_newChannels]
                    dA[i, ysh: yshk, xsw:xswk] += mat_dZ_W
                    cut = input_pd[i, ysh: yshk, xsw:xswk, :] * dZ_cut
                    dW[:, :, :, i_newChannels] += cut

    if padding == 'same':
        dA = dA[:, paddingHeight:-paddingHeight, paddingWidth:-paddingWidth, :]

    return dA, dW, db
