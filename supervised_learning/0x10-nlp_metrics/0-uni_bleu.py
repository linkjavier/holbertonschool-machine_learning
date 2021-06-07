#!/usr/bin/env python3
""" Unigram BLEU score """

import numpy as np


def uni_bleu(references, sentence):
    """ Function that calculates the unigram BLEU score for a sentence """

    unitNumber = {}

    for reference in references:
        for unit in reference:
            if unit in sentence:
                if unit not in unitNumber.keys():
                    unitNumber[unit] = reference.count(unit)
                else:
                    new = reference.count(unit)
                    actual = unitNumber[unit]
                    unitNumber[unit] = max(new, actual)

    sentenceLength = len(sentence)
    referencesList = []

    for reference in references:
        referenceLength = len(reference)
        referencesList.append(
            ((abs(referenceLength - sentenceLength)), referenceLength))

    referenceLength = sorted(referencesList, key=lambda x: x[0])
    referenceLength = referenceLength[0][1]

    if sentenceLength > referenceLength:
        bPenalty = 1
    else:
        bPenalty = np.exp(1 - (float(referenceLength) / sentenceLength))

    bleuScore = bPenalty * \
        np.exp(np.log(sum(unitNumber.values()) / sentenceLength))

    return bleuScore
