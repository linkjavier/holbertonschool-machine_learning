#!/usr/bin/env python3
""" Projection Block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Function that Builds a projection block """

    He = K.initializers.he_normal()
    F11, F3, F12 = filters
    F11Convo = K.layers.Conv2D(
        filters=F11, kernel_size=(
            1, 1), strides=(
            s, s), padding='same', kernel_initializer=He)(A_prev)
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
    SectConvo = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            s, s), padding='same', kernel_initializer=He)(A_prev)
    SectConvo = K.layers.BatchNormalization(axis=3)(SectConvo)
    Projection = K.layers.Add()([F12Convo, SectConvo])
    Projection = K.layers.Activation('relu')(Projection)
    return Projection
