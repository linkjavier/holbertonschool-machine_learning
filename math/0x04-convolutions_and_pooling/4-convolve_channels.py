#!/usr/bin/env python3
"""Convolve channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels"""
    m, h, w, c = images.shape
    KernelHeight, kernelWidth, c = kernel.shape
    StrideHeight, StrideWidth = stride

    if padding == 'valid':
        PaddingHeight = 0
        PaddingWidth = 0
    elif padding == 'same':
        PaddingHeight = int(
            (((h - 1) * StrideHeight + KernelHeight - h) / 2) + 1)
        PaddingWidth = int((((w - 1) * StrideWidth + kernelWidth - w) / 2) + 1)
    else:
        PaddingHeight, PaddingWidth = padding

    OutputH = int(((h + 2 * PaddingHeight - KernelHeight) / StrideHeight) + 1)
    OutputW = int(((w + 2 * PaddingWidth - kernelWidth) / StrideWidth) + 1)

    ImagePadded = np.pad(
        images,
        ((0, 0), (PaddingHeight, PaddingHeight),
         (PaddingWidth, PaddingWidth), (0, 0)),
        'constant'
    )

    output = np.zeros((m, OutputH, OutputW))
    ImageRange = np.arange(m)

    for i_OutputH in range(OutputH):
        for i_OutputW in range(OutputW):
            s_i_OutputH = i_OutputH * StrideHeight
            s_i_OutputW = i_OutputW * StrideWidth
            flt = ImagePadded[ImageRange,
                              s_i_OutputH:KernelHeight + s_i_OutputH,
                              s_i_OutputW:kernelWidth + s_i_OutputW,
                              :]
            output[ImageRange, i_OutputH, i_OutputW] = np.sum(
                flt * kernel, axis=(1, 2, 3))
    return output
