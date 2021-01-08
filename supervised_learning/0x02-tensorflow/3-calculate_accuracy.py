#!/usr/bin/env python3
""" Tensorflow module """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ Function that calculates the accuracy of a prediction """

    Prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

    return tf.reduce_mean(tf.cast(Prediction, tf.float32))
