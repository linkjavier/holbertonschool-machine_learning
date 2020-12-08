#!/usr/bin/env python3
"""Add 2 matrix"""


def add_matrices2D(mat1, mat2):
    """Funcxtion that add 2 matrix"""

    if (len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0])):
        columns, newMatrix = [], []
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                columns.append(mat1[i][j] + mat2[i][j])
            newMatrix.append(columns)
            columns = []
        return newMatrix
    else:
        return (None)


# import numpy as np


# def add_matrices2D(mat1, mat2):
#     if np.shape(mat1) != np.shape(mat2):
#         return None
#     return (np.add(mat1, mat2).tolist())
