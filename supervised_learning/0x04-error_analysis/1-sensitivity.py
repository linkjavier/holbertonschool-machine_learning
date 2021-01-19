#!/usr/bin/env python3
""" Sensitivity Module """
import numpy as np


def sensitivity(confusion):
    """ Function that calculates the sensitivity for each
        class in a confussion matrix
    """
    classes, _ = confusion.shape
    classSensitivity = np.zeros(classes)

    for class_ in range(classes):
        classSensitivity[class_] = np.divide(
            confusion[class_][class_], np.sum(
                confusion[class_]))

    return classSensitivity
