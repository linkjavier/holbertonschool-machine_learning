#!/usr/bin/env python3
""" Transformer Decoder Block """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ class DecoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Class constructor """
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ Method that returns a tensor of shape (batch, target_seq_len, dm)
            containing the block’s output
        """

        attention, _ = self.mha1(x, x, x, look_ahead_mask)
        attention = self.dropout1(attention, training=training)
        firstOutput = self.layernorm1(attention + x)
        secondAttention, _ = self.mha2(firstOutput,
                                       encoder_output,
                                       encoder_output,
                                       padding_mask)
        secondAttention = self.dropout2(secondAttention, training=training)
        secondOutput = self.layernorm2(secondAttention + firstOutput)
        hidden_output = self.dense_hidden(secondOutput)
        dense_output = self.dense_output(hidden_output)
        ffn_output = self.dropout3(dense_output, training=training)
        output = self.layernorm3(ffn_output + secondOutput)

        return output
