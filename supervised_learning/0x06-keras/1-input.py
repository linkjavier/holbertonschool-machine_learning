#!/usr/bin/env python3
"""Keras Neural Network Module (Input)"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Function that builds a neural network with Keras (Input)
    """
    NodesInputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    NodesOutputs = K.layers.Dense(
        units=layers[0],
        kernel_regularizer=regularizer,
        activation=activations[0],
        input_shape=(
            nx,
        ))(NodesInputs)
    for i in range(1, len(layers)):
        NodesOutputs = K.layers.Dropout(1 - keep_prob)(NodesOutputs)
        NodesOutputs = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer)(NodesOutputs)
    model = K.Model(inputs=NodesInputs, outputs=NodesOutputs)

    return model
