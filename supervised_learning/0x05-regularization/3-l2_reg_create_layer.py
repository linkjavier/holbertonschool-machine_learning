#!/usr/bin/env python3
""" Create a Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Function that creates a tensorflow layer
        that includes L2 regularizatio
    """

    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=k_init,
        kernel_regularizer=regularizer,
        activation=activation,
        name='Layer')
    output = layer(prev)
    return output
