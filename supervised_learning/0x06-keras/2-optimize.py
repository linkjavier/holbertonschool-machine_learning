#!/usr/bin/env python3
"""Keras Neural Network Module (Optimize)"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
        Function that sets up Adam optimization for a keras model with
        categorical crossentropy loss and accuracy metrics.

    """
    Optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(Optimizer, 'categorical_crossentropy',
                    ['accuracy'])
    return None
