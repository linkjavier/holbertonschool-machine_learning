#!/usr/bin/env python3
""" Precision Module """
import numpy as np


def precision(confusion):
    """ Function that calculates the sensitivity for each
        class in a confussion matrix
    """
    classes, _ = confusion.shape
    classPrecision = np.zeros(classes)

    for classItem in range(classes):
        classPrecision[classItem] = np.divide(
            confusion[classItem][classItem], np.sum(confusion[:, classItem]))

    return classPrecision
