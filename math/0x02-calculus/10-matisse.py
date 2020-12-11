#!/usr/bin/env python3
""" Module to calculate derivate """


def poly_derivative(poly):
    """ Calculates the derivative of a polynomial """

    if not poly:
        return None

    if poly:

        newCoefficient = []

        if len(poly) == 1:
            newCoefficient.append(0)

        for index in range(len(poly)):
            if index != 0:

                coefficient = poly[index]
                grade = index
                newCoefficient.append(coefficient * grade)

    else:
        return None

    return newCoefficient
