#!/usr/bin/env python3
""" matrix_shape function returns the shape of a matrix"""


def matrix_shape(matrix):
    """ Returns the shape of a matrix"""
    matrixShape = []
    if matrix:
        while (type(matrix) == list):
            matrixShape.append(len(matrix))
            matrix = matrix[0]
        return matrixShape
    else:
        return [0]

# import numpy as np


# def matrix_shape(matrix):
#     y = list(np.shape(matrix))
#     return (y)
