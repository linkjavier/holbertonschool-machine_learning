#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Function that builds an inception block """

    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    F1Convo = K.layers.Conv2D(
        filters=F1,
        kernel_size=(
            1,
            1),
        activation='relu',
        padding='same',
        kernel_initializer=init)(A_prev)
    F3RConvo = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(
            1,
            1),
        activation='relu',
        padding='same',
        kernel_initializer=init)(A_prev)
    F3Convo = K.layers.Conv2D(
        filters=F3,
        kernel_size=(
            3,
            3),
        activation='relu',
        padding='same',
        kernel_initializer=init)(F3RConvo)
    F5RConvo = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(
            1,
            1),
        activation='relu',
        padding='same',
        kernel_initializer=init)(A_prev)
    F5Convo = K.layers.Conv2D(
        filters=F5,
        kernel_size=(
            5,
            5),
        activation='relu',
        padding='same',
        kernel_initializer=init)(F5RConvo)
    MaxPool = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            1, 1), padding='same',)(A_prev)
    FPPConvo = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(
            1,
            1),
        activation='relu',
        padding='same',
        kernel_initializer=init)(MaxPool)
    ConcatOutput = K.layers.concatenate([F1Convo, F3Convo, F5Convo, FPPConvo])

    return ConcatOutput
