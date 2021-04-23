#!/usr/bin/env python3
""" 0x04. Autoencoders """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Function that creates an autoencoder """

    # Encoder
    EncoderTensor = keras.Input(shape=(input_dims,))
    EncoderLayer = keras.layers.Dense(units=hidden_layers[0], activation='relu')(EncoderTensor)

    for i in range(1, len(hidden_layers)):
        EncoderLayer = keras.layers.Dense(units=hidden_layers[i], activation='relu')(EncoderLayer)

    EncoderLayer = keras.layers.Dense(units=latent_dims, activation='relu')(EncoderLayer)

    encoder = keras.Model(inputs=EncoderTensor, outputs=EncoderLayer)

    # Decoder
    DecoderTensor = keras.Input(shape=(latent_dims,))
    DecoderLayer = keras.layers.Dense(units=hidden_layers[-1], activation='relu')(DecoderTensor)

    for i in reversed(range(len(hidden_layers) - 1)):
        DecoderLayer = keras.layers.Dense(units=hidden_layers[i], activation='relu')(DecoderLayer)

    # Last Layer of Decoder use sigmoid
    DecoderLayer = keras.layers.Dense(units=input_dims, activation='sigmoid')(DecoderLayer)

    decoder = keras.Model(inputs=DecoderTensor, outputs=DecoderLayer)

    # Autoencoder
    autoEncoderBottleneck = encoder(EncoderTensor)
    autoEncoderOutput = decoder(autoEncoderBottleneck)
    auto = keras.Model(inputs=EncoderTensor, outputs=autoEncoderOutput)

    # autoencoder model compiled using adam optimization and binary cross-entropy loss
    auto.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

    return encoder, decoder, auto
