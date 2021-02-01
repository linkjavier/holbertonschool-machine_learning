#!/usr/bin/env python3
""" Grayscale convolution padding module """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Funtion that performs a convolution on
        grayscale images with custom padding
    """

    m, h, w = images.shape
    KernelHeight, kernelWidth = kernel.shape
    PaddingHeight, PaddingWidth = padding

    OutputHeight = h + 2 * PaddingHeight - KernelHeight + 1
    OutputWidth = w + 2 * PaddingWidth - kernelWidth + 1

    ImagePadded = np.pad(
        images,
        ((0,
          0),
         (PaddingHeight,
          PaddingHeight),
            (PaddingWidth,
             PaddingWidth)),
        'constant')
    output = np.zeros((m, OutputHeight, OutputWidth))
    ImageRange = np.arange(m)

    for iOutputHeight in range(OutputHeight):
        for iOutputWidth in range(OutputWidth):
            filtered = ImagePadded[ImageRange,
                                   iOutputHeight:KernelHeight + iOutputHeight,
                                   iOutputWidth:kernelWidth + iOutputWidth]
            output[ImageRange, iOutputHeight, iOutputWidth] = np.sum(
                filtered * kernel, axis=(1, 2))
    return output
