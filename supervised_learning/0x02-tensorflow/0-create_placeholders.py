#!/usr/bin/env python3
""" Tensorflow Module """

# import tensorflow as tf
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ Function that returns two placeholders,
        x and y, for the neural network
    """
    tf.disable_v2_behavior()
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
