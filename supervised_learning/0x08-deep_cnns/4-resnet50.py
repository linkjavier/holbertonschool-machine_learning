#!/usr/bin/env python3
"""ResNet 50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds a projection block"""

    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(
            7,
            7),
        padding='same',
        strides=2,
        kernel_initializer=init)(X)

    bn1 = K.layers.BatchNormalization(axis=3)(conv1)

    activation1 = K.layers.Activation('relu')(bn1)

    maxpool1 = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding='same',)(activation1)

    Projection1 = projection_block(maxpool1, [64, 64, 256], s=1)
    IdenBlock1 = identity_block(Projection1, [64, 64, 256])
    IdenBlock2 = identity_block(IdenBlock1, [64, 64, 256])

    Projection2 = projection_block(IdenBlock2, [128, 128, 512])
    IdenBlock3 = identity_block(Projection2, [128, 128, 512])
    IdenBlock4 = identity_block(IdenBlock3, [128, 128, 512])
    IdenBlock5 = identity_block(IdenBlock4, [128, 128, 512])

    Projection3 = projection_block(IdenBlock5, [256, 256, 1024])
    IdenBlock6 = identity_block(Projection3, [256, 256, 1024])
    IdenBlock7 = identity_block(IdenBlock6, [256, 256, 1024])
    IdenBlock8 = identity_block(IdenBlock7, [256, 256, 1024])
    IdenBlock9 = identity_block(IdenBlock8, [256, 256, 1024])
    IdenBlock10 = identity_block(IdenBlock9, [256, 256, 1024])

    Projection4 = projection_block(IdenBlock10, [512, 512, 2048])
    IdenBlock11 = identity_block(Projection4, [512, 512, 2048])
    IdenBlock12 = identity_block(IdenBlock11, [512, 512, 2048])

    avgpool = K.layers.AveragePooling2D(
        pool_size=(
            1, 1), strides=(
            7, 7), padding='same',)(IdenBlock12)

    SoftMax = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )(avgpool)

    Keras = K.Model(inputs=X, outputs=SoftMax)

    return Keras
