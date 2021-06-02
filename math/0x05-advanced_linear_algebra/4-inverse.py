#!/usr/bin/env python3
""" Cofactor """


def determinant(matrix):
    """ Function that calculates the determinant of a matrix """

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))

    matrixDeterminant = []

    for i in range(len(matrix)):
        newMatrix = [[j for j in matrix[i]] for i in range(1, len(matrix))]
        for j in range(len(newMatrix)):
            newMatrix[j].pop(i)
        if i % 2 == 0:
            matrixDeterminant.append(matrix[0][i] * determinant(newMatrix))
        if i % 2 == 1:
            matrixDeterminant.append(-1 *
                                     matrix[0][i] * determinant(newMatrix))

    return sum(matrixDeterminant)


def minor(matrix):
    """Function that calculates the minor matrix of a matrix"""

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if not isinstance(i, list):
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

    if not isinstance(matrix, list) or len(matrix) is 0:
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

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if not isinstance(i, list):
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


def inverse(matrix):
    """ Function that calculates the inverse of a matrix """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")

    matrixLength = len(matrix)
    if matrixLength == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for element in matrix:
        if len(element) != matrixLength:
            raise ValueError("matrix must be a non-empty square matrix")

    matrixDeterminant = determinant(matrix)

    if matrixDeterminant == 0:
        return None

    matrixAdjugate = adjugate(matrix)

    inverseMatrix = [[n / matrixDeterminant for n in row]
                     for row in matrixAdjugate]

    return inverseMatrix
