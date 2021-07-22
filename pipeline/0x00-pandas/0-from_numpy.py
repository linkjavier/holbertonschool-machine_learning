#!/usr/bin/env python3
""" From Numpy """

import pandas as pd
import numpy as np


def from_numpy(array):
    """ Function that creates a pd.DataFrame from a np.ndarray  """

    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVW")
    columns = array.shape[1]

    return pd.DataFrame(array, columns=alphabet[:columns])
