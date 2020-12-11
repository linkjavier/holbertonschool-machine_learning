#!/usr/bin/env python3
""" Module to calculate derivate """


def poly_derivative(poly):
    """ Calculates the derivative of a polynomial """

    if type(poly) != list or len(poly) == 0:
        return None

    # if len(poly) == 1:
    #     return [0]

    if poly:

        result = []

        if len(poly) == 1:
            result.append(0)

        for index in range(len(poly)):
            if index != 0:

                coefficient = poly[index]
                grade = index
                result.append(coefficient * grade)

    else:
        return None

    return result
