#!/usr/bin/env python3
"""  Momentum Upgraded Module"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ Function that creates the training operation for a neural network
        using gradient descent with momentum optimization algorithm
    """

    train = tf.train.MomentumOptimizer(alpha, beta1)
    NewGradients = train.compute_gradients(loss)
    momentumOptimization = train.apply_gradients(NewGradients)

    return momentumOptimization
