#!/usr/bin/env python3
""" Tensorflow module """

import tensorflow as tf


def create_train_op(loss, alpha):
    """ Function that creates the training operation for the network """

    train = tf.train.GradientDescentOptimizer(alpha)
    gradient = train.compute_gradients(loss)

    return train.apply_gradients(gradient)
