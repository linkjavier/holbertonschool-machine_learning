#!/usr/bin/env python3
"""  """
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """ class Encoder that inherits from tensorflow.keras.layers.Layer
        to create the encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """ Class constructor """

        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """ Method that returns tensor of shape (batch, input_seq_len, dm) """

        input_seq_len = x.shape[1]
        embeddings = self.embedding(x)
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embeddings += self.positional_encoding[:input_seq_len]
        output = self.dropout(embeddings, training=training)

        for i in range(self.N):
            output = self.blocks[i](output, training, mask)

        return output
