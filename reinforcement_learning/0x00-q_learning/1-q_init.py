#!/usr/bin/env python3
""" Initialize Q-table """
import numpy as np


def q_init(env):
    """ Function that initializes the Q-table """

    ActionSpace = env.action_space.n
    StateSpace = env.observation_space.n
    Qtable = np.zeros((StateSpace, ActionSpace))

    return Qtable
