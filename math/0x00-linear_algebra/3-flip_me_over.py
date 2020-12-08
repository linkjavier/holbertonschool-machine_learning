#!/usr/bin/env python3
""" The matrix_transpose """


def matrix_transpose(m):
    """Returns the transpose of matrix 2D"""
    transposeMatrix = []
    rows = len(m)
    columns = len(m[0])
    if not isinstance(m, list) or not len(m) > 0 or not isinstance(m[0], list):
        return None
    for i in range(columns):
        transposeMatrix.append([m[r][i] for r in range(rows)])
    return transposeMatrix


# import numpy as np


# def matrix_transpose(matrix):
#     y = np.transpose(matrix).tolist()
#     return (y)
