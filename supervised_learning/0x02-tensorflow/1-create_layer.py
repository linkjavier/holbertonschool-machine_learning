#!/usr/bin/env python3
"""Tendsorflow module"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Function that creates a layer """

    Initialization = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=Initialization,
        activation=activation,
        name='Layer')
    OutputLayer = layer(prev)
    return OutputLayer
