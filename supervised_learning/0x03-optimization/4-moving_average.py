#!/usr/bin/env python3
"""Moving module"""


def moving_average(data, beta):
    """ Function  that calculates the weighted
        moving average of a data set
    """
    weightedMovingAverage = list()
    epsilon = 1 - beta
    NewWeightedValue = 0

    for i, theta in enumerate(data, start=1):
        bias_correction = 1 - (beta ** i)
        NewWeightedValue = (beta * NewWeightedValue) + (epsilon * theta)
        weightedMovingAverage.append(NewWeightedValue / bias_correction)

    return weightedMovingAverage
