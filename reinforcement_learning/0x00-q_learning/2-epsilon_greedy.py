#!/usr/bin/env python3
""" Epsilon Greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Function that uses epsilon-greedy to determine the next action """

    e = np.random.uniform(0, 1)

    if e > epsilon:
        NextActionIndex = np.argmax(Q[state, :])
    else:
        NextActionIndex = np.random.randint(0, 3, None)

    return NextActionIndex

