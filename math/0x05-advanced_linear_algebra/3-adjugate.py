#!/usr/bin/env python3
""" Cofactor """


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


def cofactor(matrix):
    """ Function that calculates the cofactor matrix of a matrix """

    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

    cofactorMatrix = []

    for i in range(len(matrix)):
        cofactorMatrix.append([])
        for j in range(len(matrix)):
            sign = (-1) ** (i + j)
            element = sign * minor(matrix)[i][j]
            cofactorMatrix[i].append(element)

    return cofactorMatrix


def adjugate(matrix):
    """ Function that calculates the adjugate matrix of a matrix """

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

    adjugate = cofactor(matrix)

    newMatrix = [[j for j in adjugate[i]] for i in range(len(adjugate))]

    for i in range(len(adjugate)):
        for j in range(len(adjugate[i])):
            adjugate[j][i] = newMatrix[i][j]

    return adjugate
