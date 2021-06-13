#!/usr/bin/env python3
""" Transformer Decoder """

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ Class Decoder that inherits from tensorflow.keras.layers.Layer
        to create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """ Class constructor """

        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """ Method that returns tensor of shape
            (batch, target_seq_len, dm)
        """

        target_seq_len = x.shape[1]
        embeddings = self.embedding(x)
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embeddings += self.positional_encoding[:target_seq_len]
        output = self.dropout(embeddings, training=training)

        for i in range(self.N):
            output = self.blocks[i](
                output,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask)

        return output
