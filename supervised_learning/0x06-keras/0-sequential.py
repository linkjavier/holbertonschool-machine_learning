#!/usr/bin/env python3
"""Keras Neural Network Module"""
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        Function that builds a neural network with Keras
    """
    model = tf.keras.Sequential()
    regularizer = tf.keras.regularizers.l2(lambtha)
    model.add(
        tf.keras.layers.Dense(
            units=layers[0],
            activation=activations[0],
            kernel_regularizer=regularizer,
            input_shape=(
                nx,
            )))

    for i in range(1, len(layers)):
        model.add(tf.keras.layers.Dropout(1 - keep_prob))
        model.add(
            tf.keras.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer))
    return model
