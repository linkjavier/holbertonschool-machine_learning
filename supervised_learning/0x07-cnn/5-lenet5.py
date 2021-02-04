#!/usr/bin/env python3
""" LeNet-5 (Keras) """
import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of
        the LeNet-5 architecture using keras
    """

    init = K.initializers.he_normal()

    ConvoLayer1 = K.layers.Conv2D(
        6, (5, 5), activation='relu', padding='same',
        kernel_initializer=init)(X)

    maxpoolLayer1 = K.layers.MaxPooling2D((2, 2), (2, 2),)(ConvoLayer1)

    ConvoLayer2 = K.layers.Conv2D(
        16,
        (5,
         5),
        activation='relu',
        padding='valid',
        kernel_initializer=init)(maxpoolLayer1)

    maxpoolLayer2 = K.layers.MaxPooling2D((2, 2), (2, 2),)(ConvoLayer2)

    flatten = K.layers.Flatten()(maxpoolLayer2)

    FullyConnectedLayer1 = K.layers.Dense(
        120, activation='relu', kernel_initializer=init)(flatten)

    FullyConnectedLayer2 = K.layers.Dense(
        84, activation='relu', kernel_initializer=init)(FullyConnectedLayer1)

    FullyConnectedLayer3 = K.layers.Dense(
        10, activation='softmax',
        kernel_initializer=init)(FullyConnectedLayer2)

    model = K.Model(inputs=X, outputs=FullyConnectedLayer3)

    AdamOptimization = K.optimizers.Adam()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=AdamOptimization,
        metrics=['accuracy'])

    return model
