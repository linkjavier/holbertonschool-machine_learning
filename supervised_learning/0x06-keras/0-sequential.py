#!/usr/bin/env python3
"""Keras Neural Network Module"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Function that builds a neural network with Keras
    """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)
    model.add(
        K.layers.Dense(
            units=layers[0],
            activation=activations[0],
            kernel_regularizer=regularizer,
            input_shape=(
                nx,
            )))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(
            K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer))

    return model
