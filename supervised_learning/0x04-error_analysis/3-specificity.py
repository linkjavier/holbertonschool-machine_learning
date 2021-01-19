#!/usr/bin/env python3
""" Specificity Module """
import numpy as np


def specificity(confusion):
    """Function that calculates the specificity for each
        class in a confusion matrix
    """
    classes, _ = confusion.shape
    classSpecificity = np.zeros(classes)

    total = np.sum(confusion)

    for classItem in range(classes):
        truePositive = confusion[classItem][classItem]
        falsePositive = np.sum(confusion[classItem]) - truePositive
        falseNegative = np.sum(confusion[:, classItem]) - truePositive

        subTotal = total - falsePositive - falseNegative - truePositive

        classSpecificity[classItem] = np.divide(
            subTotal, subTotal + falseNegative)

    return classSpecificity
