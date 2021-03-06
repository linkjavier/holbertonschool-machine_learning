#!/usr/bin/env python3
"""Dense block"""
import tensorflow.keras as K


def dense_block(X, FilterNumber, growth_rate, layers):
    """Builds a Dense block"""

    init = K.initializers.he_normal(seed=None)

    for _ in range(layers):
        Normal1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(Normal1)
        FirstConvo = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=init
        )(activation1)
        Normal2 = K.layers.BatchNormalization()(FirstConvo)
        activation2 = K.layers.Activation('relu')(Normal2)
        SecondConvo = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=init
        )(activation2)
        X = K.layers.concatenate([X, SecondConvo])
        FilterNumber += growth_rate

    return X, FilterNumber
