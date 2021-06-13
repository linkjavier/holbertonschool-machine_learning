#!/usr/bin/env python3
""" RNN Decoder """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor """

        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """ Method that calls the tensors containing the output word
            and the new decoder hidden state
        """

        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, _ = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)
        embeddings = tf.concat([tf.expand_dims(context, 1), embeddings], -1)
        output, s = self.gru(embeddings)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, s
