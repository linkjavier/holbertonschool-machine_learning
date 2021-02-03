#!/usr/bin/env python3
""" Convolutional Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Function that performs forward propagation over a
        convolutional layer of a neural network
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

    output = np.zeros((m, outputH, outputW, newChannels))
    rng_im = np.arange(m)

    for k in range(newChannels):
        for i_outputH in range(outputH):
            for i_outputW in range(outputW):
                s_i_outputH = i_outputH * strideHeight
                s_i_outputW = i_outputW * strideWidth
                flt = input_pd[rng_im,
                               s_i_outputH:kernelHeight + s_i_outputH,
                               s_i_outputW:kernelWidth + s_i_outputW]
                kernel = W[:, :, :, k]
                output[rng_im, i_outputH, i_outputW, k] = np.sum(
                    flt * kernel, axis=(1, 2, 3)
                )
    finalOutput = output + b
    return activation(finalOutput)
