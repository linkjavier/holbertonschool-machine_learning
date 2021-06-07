#!/usr/bin/env python3
""" Cumulative N-gram BLEU score """
import numpy as np


def gramBuilder(sentence, n):
    """ Function that builds grams from sentence """

    candidates = []

    for i in range(len(sentence)):
        last = i + n
        begin = i
        if last >= len(sentence) + 1:
            break

        gram = sentence[begin: last]
        element = ' '.join(gram)
        candidates.append(element)

    return candidates


def ngram_bleu(references, sentence, n):
    """ Function that calculates the n-gram BLEU score for a sentence: """

    wordsNumber = {}
    candidates = gramBuilder(sentence, n)
    candidatesLength = len(candidates)

    gramUnits = []

    for reference in references:
        gramsList = gramBuilder(reference, n)
        gramUnits.append(gramsList)

    for units in gramUnits:
        for word in units:
            if word in candidates:
                if word not in wordsNumber.keys():
                    wordsNumber[word] = units.count(word)
                else:
                    actual = units.count(word)
                    before = wordsNumber[word]
                    wordsNumber[word] = max(actual, before)

    probability = sum(wordsNumber.values()) / candidatesLength

    return probability


def cumulative_bleu(references, sentence, n):
    """ Function that calculates the cumulative n-gram BLEU score
        for a sentence
    """

    perplex = []

    for i in range(1, n+1):
        result = ngram_bleu(references, sentence, i)
        perplex.append(result)

    tuplesMatch = []

    for reference in references:
        referenceLength = len(reference)
        refLengthDiff = abs(referenceLength - len(sentence))
        tuplesMatch.append((refLengthDiff, referenceLength))

    tuplesMatchSorted = sorted(tuplesMatch, key=lambda x: x[0])
    bestMatch = tuplesMatchSorted[0][1]

    if len(sentence) > bestMatch:
        bPenalty = 1
    else:
        bPenalty = np.exp(1 - (bestMatch / len(sentence)))

    bleuScore = bPenalty * np.exp(np.sum(np.log(perplex)) / n)

    return bleuScore
