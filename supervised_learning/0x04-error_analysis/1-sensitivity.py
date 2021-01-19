#!/usr/bin/env python3
""" Sensitivity Module """
import numpy as np


def sensitivity(confusion):
    """ Function that calculates the sensitivity for each
        class in a confussion matrix
    """
    classes, _ = confusion.shape
    classSensitivity = np.zeros(classes)

    for classItem in range(classes):
        classSensitivity[classItem] = np.divide(
            confusion[classItem][classItem], np.sum(
                confusion[classItem]))

    return classSensitivity
