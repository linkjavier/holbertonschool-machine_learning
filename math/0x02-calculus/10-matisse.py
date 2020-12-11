#!/usr/bin/env python3
""" Module to calculate derivate """


def poly_derivative(poly):
    """ Calculates the derivative of a polynomial """

    polyType = type(poly)
    polyTypeFirst = type(poly[0])
    lenPoly = len(poly)
    result = []
    zeroFlag = True

    if polyType != list or poly == [] or polyTypeFirst not in [int, float]:
        return None

    if lenPoly == 1:
        return [0]

    for index, coeficient in enumerate(poly[1:]):
        if type(coeficient) not in [int, float]:
            return None

        result.append((index + 1) * coeficient)
        if zeroFlag and result[-1]:
            zeroFlag = False

    if zeroFlag:
        return [0]

    return result
