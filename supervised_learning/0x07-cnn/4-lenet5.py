#!/usr/bin/env python3
""" LeNet-5 (Tensorflow) """
import tensorflow as tf


def lenet5(x, y):
    """ Function that builds a modified version of the
        LeNet-5 architecture using tensorflow
    """

    init = tf.contrib.layers.variance_scaling_initializer()

    activation = tf.nn.relu

    ConvoLayer1 = tf.layers.Conv2D(
        6, (5, 5), activation=activation, padding='same',
        kernel_initializer=init)(x)

    maxpoolLayer1 = tf.layers.MaxPooling2D((2, 2), (2, 2),)(ConvoLayer1)

    ConvoLayer2 = tf.layers.Conv2D(
        16,
        (5,
         5),
        activation=activation,
        padding='valid',
        kernel_initializer=init)(maxpoolLayer1)

    maxpoolLayer2 = tf.layers.MaxPooling2D((2, 2), (2, 2),)(ConvoLayer2)

    flatten = tf.layers.Flatten()(maxpoolLayer2)

    FullyConnectedLayer1 = tf.layers.Dense(
        120, activation=activation, kernel_initializer=init)(flatten)

    FullyConnectedLayer2 = tf.layers.Dense(
        84, activation=activation,
        kernel_initializer=init)(FullyConnectedLayer1)

    FullyConnectedLayer3 = tf.layers.Dense(
        10, kernel_initializer=init)(FullyConnectedLayer2)

    loss = tf.losses.softmax_cross_entropy(y, FullyConnectedLayer3)

    AdamTrain = tf.train.AdamOptimizer().minimize(loss)

    prediction = tf.equal(tf.argmax(FullyConnectedLayer3, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    softmaxActivatedOutput = tf.nn.softmax(FullyConnectedLayer3)

    return softmaxActivatedOutput, AdamTrain, loss, accuracy
