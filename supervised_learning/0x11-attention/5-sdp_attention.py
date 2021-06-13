#!/usr/bin/env python3
""" Scaled Dot Product Attention """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ Function that calculates the scaled dot product attention """

    productMatrix = tf.matmul(Q, K, transpose_b=True)
    floatTensor = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = productMatrix / tf.math.sqrt(floatTensor)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
