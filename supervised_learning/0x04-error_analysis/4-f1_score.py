#!/usr/bin/env python3
""" F1 Score Module """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Function that calculates the F1 score
        of a confusion matrix
    """
    LoadedSensitivity = sensitivity(confusion)
    LoadedPrecision = precision(confusion)

    classF1Score = np.divide(2, np.power(
        LoadedSensitivity, -1) + np.power(LoadedPrecision, -1))

    return classF1Score
