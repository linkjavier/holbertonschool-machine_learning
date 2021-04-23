#!/usr/bin/env python3
""" Variational Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Function that creates a variational autoencoder """

    def vaeLoss(data):
        """ Function that vaeLoss Layer  executes """

        mean, logSigma = data
        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        vaeLoss = mean + keras.backend.exp(logSigma / 2) * epsilon

        return vaeLoss

    def KullbackLeibler(Any, Any2):
        """ Kullback-Leibler divergence Loss reconstructor """

        loss = keras.losses.binary_crossentropy(EncoderTensor,
                                                autoEncoderOutput)
        loss *= input_dims
        exp = keras.backend.exp(logSigma)
        KullbackLoss = 1 + logSigma - keras.backend.square(mean) - exp
        KullbackLoss = keras.backend.sum(KullbackLoss, axis=-1)
        KullbackLoss *= -0.5
        newLoss = keras.backend.mean(loss + KullbackLoss)

        return newLoss

    # Encoder
    EncoderTensor = keras.layers.Input(shape=(input_dims,))
    EncoderLayer = EncoderTensor

    for layer in hidden_layers:
        EncoderLayer = keras.layers.Dense(
            layer,
            activation='relu')(EncoderLayer)

    mean = keras.layers.Dense(latent_dims)(EncoderLayer)
    logSigma = keras.layers.Dense(latent_dims)(EncoderLayer)
    vaeLossLayer = keras.layers.Lambda(vaeLoss)([mean, logSigma])

    encoder = keras.Model(EncoderTensor, [vaeLossLayer, mean, logSigma])

    # Decoder
    DecoderTensor = keras.layers.Input(shape=(latent_dims,))
    DecoderLayer = DecoderTensor

    for layer in reversed(hidden_layers):
        DecoderLayer = keras.layers.Dense(
            layer,
            activation='relu')(DecoderLayer)

    DecoderLayer = keras.layers.Dense(
        input_dims, activation='sigmoid')(DecoderLayer)

    decoder = keras.Model(inputs=DecoderTensor, outputs=DecoderLayer)

    # Autoencoder
    autoEncoderBottleneck = encoder(EncoderTensor)
    autoEncoderOutput = decoder(autoEncoderBottleneck)
    auto = keras.models.Model(EncoderTensor, autoEncoderOutput)

    auto.compile(optimizer='adam', loss=KullbackLeibler)

    return encoder, decoder, auto
