#!/usr/bin/env python3
""" Positional Encoding """

import numpy as np


def positional_encoding(max_seq_len, dm):
    """ Function that calculates the positional encoding for a transformer """

    PEV = np.zeros((max_seq_len, dm))

    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            expQuotient = np.exp(j * -np.log(10000.0) / dm)
            PEV[i, j] = np.sin(i * expQuotient)
            PEV[i, j + 1] = np.cos(i * expQuotient)

    return PEV
