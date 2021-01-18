#!/usr/bin/env python3
""" Create confusion module """
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix:"""

    m, classes = labels.shape
    ConfusionMatrix = np.zeros((classes, classes))

    for i in range(m):
        Indexlabel = labels[i]
        Indexlogit = logits[i]

        x, *_ = np.where(Indexlabel == 1)
        y, *_ = np.where(Indexlogit == 1)

        ConfusionMatrix[x[0]][y[0]] += 1

    return ConfusionMatrix
