#!/usr/bin/env python3
""" ActivatedOutputtity Block  """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Function that builds an ActivatedOutputtity block """
    He = K.initializers.he_normal()
    F11, F3, F12 = filters
    F11Convo = K.layers.Conv2D(filters=F11, kernel_size=(
        1, 1), padding='same', kernel_initializer=He)(A_prev)
    F11Convo = K.layers.BatchNormalization(axis=3)(F11Convo)
    F11Convo = K.layers.Activation('relu')(F11Convo)
    F3Convo = K.layers.Conv2D(
        filters=F3,
        kernel_size=(
            3,
            3),
        padding='same',
        kernel_initializer=He)(F11Convo)
    F3Convo = K.layers.BatchNormalization(axis=3)(F3Convo)
    F3Convo = K.layers.Activation('relu')(F3Convo)
    F12Convo = K.layers.Conv2D(filters=F12, kernel_size=(
        1, 1), padding='same', kernel_initializer=He)(F3Convo)
    F12Convo = K.layers.BatchNormalization(axis=3)(F12Convo)
    ActivatedOutput = K.layers.Add()([F12Convo, A_prev])
    ActivatedOutput = K.layers.Activation('relu')(ActivatedOutput)
    return ActivatedOutput
