#!/usr/bin/env python3
import numpy as np


def add_arrays(arr1, arr2):
    if np.shape(arr1) != np.shape(arr2):
        return None
    return (list(np.add(arr1, arr2)))
