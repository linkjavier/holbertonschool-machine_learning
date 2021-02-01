#!/usr/bin/env python3
""" Grayscale convolution module """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Function that performs a convolution
        on grayscale images.
    """
    m, h, w = images.shape
    KernelHeight, kernelWidth = kernel.shape

    if KernelHeight % 2 == 0:
        PaddingHeight = int(KernelHeight / 2)
    else:
        PaddingHeight = int((KernelHeight - 1) / 2)

    if kernelWidth % 2 == 0:
        PaddingWidth = int(kernelWidth / 2)
    else:
        PaddingWidth = int((kernelWidth - 1) / 2)

    PaddedImage = np.pad(
        images,
        ((0,
          0),
         (PaddingHeight,
          PaddingHeight),
            (PaddingWidth,
             PaddingWidth)),
        'constant')

    output = np.zeros((m, h, w))
    RangeImages = np.arange(m)

    for i_h in range(h):
        for i_w in range(w):
            filtered = PaddedImage[RangeImages,
                                   i_h:KernelHeight + i_h,
                                   i_w:kernelWidth + i_w]
            output[RangeImages, i_h, i_w] = np.sum(
                filtered * kernel, axis=(1, 2))
    return output
