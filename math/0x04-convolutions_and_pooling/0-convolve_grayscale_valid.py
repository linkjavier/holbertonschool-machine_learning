#!/usr/bin/env python3
""" Grayscale convolution module """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Function that performs a valid
        convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    oh = h - kh + 1
    ow = w - kw + 1

    Convolved = np.zeros((m, oh, ow))
    ImageRange = np.arange(m)

    for i_oh in range(oh):
        for i_ow in range(ow):
            filtered = images[ImageRange, i_oh:kh + i_oh, i_ow:kw + i_ow]
            Convolved[ImageRange, i_oh, i_ow] = np.sum(
                filtered * kernel, axis=(1, 2))
    return Convolved
