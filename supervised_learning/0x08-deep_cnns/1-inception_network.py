#!/usr/bin/env python3
""" Inception network """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds an inception network"""

    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    FirstConvo = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(
        2, 2), activation='relu', padding='same', kernel_initializer=init)(X)
    FirstMaxPool = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding='same',)(FirstConvo)
    SecondConvo = K.layers.Conv2D(
        filters=64,
        kernel_size=(
            1,
            1),
        strides=(
            1,
            1),
        activation='relu',
        padding='same',
        kernel_initializer=init)(FirstMaxPool)
    ThirdConvo = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', padding='same',
        kernel_initializer=init)(SecondConvo)
    SecondMaxPool = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding='same',)(ThirdConvo)
    inception3a = inception_block(SecondMaxPool, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])
    ThirdMaxPool = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding='same',)(inception3b)
    inception4a = inception_block(ThirdMaxPool, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])
    FourthMaxPool = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding='same',)(inception4e)
    inception5a = inception_block(FourthMaxPool, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])
    AveragePool = K.layers.AveragePooling2D(
        pool_size=(
            1, 1), strides=(
            7, 7), padding='same',)(inception5b)
    Dropout = K.layers.Dropout(0.4)(AveragePool)
    softmax = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )(Dropout)
    KerasModel = K.Model(inputs=X, outputs=softmax)

    return KerasModel
