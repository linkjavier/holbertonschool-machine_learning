#!/usr/bin/env python3
"""SARSA(λ) Method"""

import gym
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ Function that performs SARSA(λ) """

    def epsilon_greedy(state, Q, epsilon):
        """ Eplison greedy method """

        p = np.random.uniform(0, 1)
        if p >= epsilon:
            action = np.argmax(Q[state])
        else:
            action = np.random.randint(0, int(Q.shape[1]))

        return(action)

    stateSpaceSize = env.observation_space.n
    eps = epsilon
    Et = np.zeros((Q.shape))

    for episode in range(episodes):
        s = env.reset()
        action = epsilon_greedy(s, Q, epsilon=epsilon)

        for _ in range(max_steps):
            Et = Et * lambtha * gamma
            Et[s, action] += 1.0
            new_s, reward, done, _ = env.step(action)
            new_a = epsilon_greedy(new_s, Q, epsilon=epsilon)
            if env.desc.reshape(stateSpaceSize)[new_s] == b'H':
                reward = -1.0
            if env.desc.reshape(stateSpaceSize)[new_s] == b'G':
                reward = 1.0
            delta_t = reward + gamma * Q[new_s, new_a] - Q[s, action]
            Q[s, action] = Q[s, action] + alpha * delta_t * Et[s, action]
            if done:
                break
            s = new_s
            action = new_a

        epsilon = min_epsilon + (eps - min_epsilon) *\
            np.exp(-epsilon_decay * episode)

    return Q
