#!/usr/bin/env python3
""" Convolutional Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ Function that creates an convolutional autoencoder """

    # Encoder
    EncoderTensor = keras.Input(shape=(input_dims))
    EncoderLayer = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=3,
        padding='same',
        activation='relu')(EncoderTensor)

    EncoderLayer = keras.layers.MaxPool2D(
        pool_size=2,
        padding='same')(EncoderLayer)

    for i in range(1, len(filters)):
        EncoderLayer = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=3,
            padding='same',
            activation='relu')(EncoderLayer)
        EncoderLayer = keras.layers.MaxPool2D(
            pool_size=2,
            padding='same')(EncoderLayer)

    encoder = keras.Model(inputs=EncoderTensor, outputs=EncoderLayer)

    # Decoder
    DecoderTensor = keras.Input(shape=(latent_dims))
    DecoderLayer = keras.layers.Conv2D(
        filters=filters[-1],
        kernel_size=3,
        padding='same',
        activation='relu')(DecoderTensor)
    DecoderLayer = keras.layers.UpSampling2D(size=2)(DecoderLayer)

    for i in reversed(range(2, len(filters))):
        DecoderLayer = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=3,
            padding='same',
            activation='relu')(DecoderLayer)
        DecoderLayer = keras.layers.UpSampling2D(size=2)(DecoderLayer)

    DecoderLayer = keras.layers.Conv2D(
        filters=input_dims[0],
        kernel_size=3,
        padding='valid',
        activation='relu')(DecoderLayer)

    DecoderLayer = keras.layers.UpSampling2D(size=2)(DecoderLayer)

    DecoderLayer = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=3,
        padding='same',
        activation='sigmoid')(DecoderLayer)

    decoder = keras.Model(inputs=DecoderTensor, outputs=DecoderLayer)

    # Autoencoder
    autoEncoderBottleneck = encoder(EncoderTensor)
    autoEncoderOutput = decoder(autoEncoderBottleneck)
    auto = keras.Model(inputs=EncoderTensor, outputs=autoEncoderOutput)

    # autoencoder model compiled using adam optimization and binary
    # cross-entropy loss
    auto.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    return encoder, decoder, auto
