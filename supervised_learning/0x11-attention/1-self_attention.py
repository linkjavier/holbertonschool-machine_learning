#!/usr/bin/env python3
""" Self Attention """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to calculate the attention for machine translation
    """

    def __init__(self, units):
        """ Class constructor """

        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ Method that calls the context vector and the attention weights
            of the encoder
        """

        s_prev = tf.expand_dims(s_prev, 1)
        e = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(e, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
