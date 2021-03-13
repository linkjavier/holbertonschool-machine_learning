#!/usr/bin/env python3
"""
This module contains the function mat_mul.
"""


def mat_mul(mat1, mat2):
    """
    Function to multyply 2 2D matrices.
    """
    rows1 = len(mat1)
    columns1 = len(mat1[0])
    rows2 = len(mat2)
    columns2 = len(mat2[0])
    if columns1 != rows2:
        return None
    ans = []
    for i in range(rows1):
        current = []
        for col in range(columns2):
            element = 0
            for j in range(columns1):
                element += (mat1[i][j] * mat2[j][col])
            current.append(element)
        ans.append(current)
    return ans
