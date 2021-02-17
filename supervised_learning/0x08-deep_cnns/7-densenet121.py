#!/usr/bin/env python3
""" DenseNet121 """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet121"""

    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    Normal1 = K.layers.BatchNormalization()(X)
    FirstActivation = K.layers.Activation('relu')(Normal1)

    FirstConvo = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=2,
        kernel_initializer=init
    )(FirstActivation)

    MaxPool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )(FirstConvo)

    DenseBlock1, f1 = dense_block(MaxPool1, 2 * growth_rate, growth_rate, 6)
    FirstTransition, f2 = transition_layer(DenseBlock1, f1, compression)
    DenseBlock2, f3 = dense_block(FirstTransition, f2, growth_rate, 12)
    SecondTransition, f4 = transition_layer(DenseBlock2, f3, compression)
    DenseBlock3, f5 = dense_block(SecondTransition, f4, growth_rate, 24)
    ThridTransition, f6 = transition_layer(DenseBlock3, f5, compression)
    DenseBlock4, f7 = dense_block(ThridTransition, f6, growth_rate, 16)
    avgpool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='same',
    )(DenseBlock4)
    softmax = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )(avgpool)
    Keras = K.Model(inputs=X, outputs=softmax)

    return Keras
