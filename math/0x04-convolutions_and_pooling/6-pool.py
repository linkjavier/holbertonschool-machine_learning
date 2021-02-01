#!/usr/bin/env python3
"""Pool Convolve Module"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ Function that performs pooling on images """
    m, h, w, c = images.shape
    KernelHeight, kernelWidth = kernel_shape
    StrideHeight, StrideWidth = stride

    OutputH = int(((h - KernelHeight) / StrideHeight) + 1)
    OutputW = int(((w - kernelWidth) / StrideWidth) + 1)

    output = np.zeros((m, OutputH, OutputW, c))
    ImageRange = np.arange(m)

    for i_OutputH in range(OutputH):
        for i_OutputW in range(OutputW):
            s_i_OutputH = i_OutputH * StrideHeight
            s_i_OutputW = i_OutputW * StrideWidth
            flt = images[ImageRange, s_i_OutputH: KernelHeight +
                         s_i_OutputH, s_i_OutputW: kernelWidth + s_i_OutputW]
            if mode == 'max':
                output[ImageRange, i_OutputH, i_OutputW] = flt.max(axis=(1, 2))
            elif mode == 'avg':
                output[ImageRange, i_OutputH, i_OutputW] = np.mean(
                    flt, axis=(1, 2))
    return output
