#!/usr/bin/env python3
""" Module to calculate derivate """


def poly_derivative(poly):
    """ Calculates the derivative of a polynomial """

    lenPoly = len(poly)
    result = []
    zeroFlag = True

    if type(poly) != list or poly == [] or type(poly[0]) not in [int, float]:
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
