#!/usr/bin/env python3
""" Minor """


def determinant(matrix):
    """ Function that calculates the determinant of a matrix """

    if len(matrix) == 2:
        simpleDeterminant = (matrix[0][0] * matrix[1]
                             [1]) - (matrix[0][1] * matrix[1][0])
        return simpleDeterminant

    answerDeterminant = 0

    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        auxMatrix = [[row[n]
                      for n in range(len(matrix)) if n != i] for row in rows]
        answerDeterminant += k * (-1) ** i * determinant(auxMatrix)

    return answerDeterminant


def minor(matrix):
    """Function that calculates the minor matrix of a matrix"""

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if len(matrix) != len(i):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        minor = [i[::-1] for i in matrix]
        return minor[::-1]

    minor = [[j for j in matrix[i]] for i in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            newMatrix = [[j for j in matrix[i]] for i in range(len(matrix))]
            newMatrix = newMatrix[:i] + newMatrix[i + 1:]
            for k in range(len(newMatrix)):
                newMatrix[k].pop(j)
            minor[i][j] = determinant(newMatrix)

    return minor
