#!/usr/bin/env python3
""" Monte Carlo """

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """  Function that performs the Monte Carlo algorithm """

    for i in range(episodes):
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            episode.append([state, action, reward, new_state])
            if done:
                break
            state = new_state

        episode = np.array(episode, dtype=int)
        G = 0

        for _, step in enumerate(episode[::-1]):
            state, action, reward, _ = step
            G = gamma * G + reward
            if state not in episode[:i, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
