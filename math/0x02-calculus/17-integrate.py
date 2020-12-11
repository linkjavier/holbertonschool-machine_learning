#!/usr/bin/env python3
""" Module to integrate functions """


def poly_integral(poly, C=0):
    """  Function that calculates the integral of a polynomial """

    polyType = type(poly)
    result = []

    if polyType != list or poly == [] or type(C) not in [int, float]:
        return None

    integrationConstant = int(C)
    if integrationConstant == C:
        result.append(integrationConstant)
    else:
        result.append(C)

    for index, coefficient in enumerate(poly):
        if type(coefficient) not in [int, float]:
            return None

        new = coefficient / (index + 1)
        integrationConstant = int(new)

        if integrationConstant == new:
            result.append(integrationConstant)
        else:
            result.append(new)

    while result and result[-1] == 0:
        result.pop()

    if not result:
        return [0]

    return result
