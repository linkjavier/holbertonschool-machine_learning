#!/usr/bin/env python3
"""Transition Layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a Transition Layer"""

    init = K.initializers.he_normal(seed=None)

    Normal1 = K.layers.BatchNormalization()(X)
    FirstActivation = K.layers.Activation('relu')(Normal1)
    filters = int(nb_filters * compression)
    FirstConvo = K.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(FirstActivation)

    avgpool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
    )(FirstConvo)

    return avgpool, filters
