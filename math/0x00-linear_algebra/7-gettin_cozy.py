#!/usr/bin/env python3
""" cat_matrices2D """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Function to concatenate 2D matrices over an axis """

    rows = len(mat2)
    columns = len(mat2[0])
    if (axis == 0):
        if (columns != len(mat1[0])):
            return None
        newMatrix = []

        for row in mat1:
            newMatrix.append(row.copy())
        return newMatrix + mat2

    elif (axis == 1):
        if (rows != len(mat1)):
            return None
        newMatrix = []

        for i in range(rows):
            newMatrix.append(mat1[i] + mat2[i])

        return newMatrix

    return None

# #!/usr/bin/env python3
# import numpy as np


# def cat_matrices2D(mat1, mat2, axis=0):
#     return ((np.concatenate((mat1, mat2), axis)).tolist())
