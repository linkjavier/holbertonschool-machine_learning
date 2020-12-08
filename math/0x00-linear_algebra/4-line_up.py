#!/usr/bin/env python3
"""add_arrays function"""


def add_arrays(array1, array2):
    """ Function that adds two arrays element-wise """

    if len(array1) == len(array2):
        return [array1[i] + array2[i] for i in range(0, len(array1), 1)]
    return None


# #!/usr/bin/env python3
# import numpy as np


# def add_arrays(arr1, arr2):
#     if np.shape(arr1) != np.shape(arr2):
#         return None
#     return (list(np.add(arr1, arr2)))
