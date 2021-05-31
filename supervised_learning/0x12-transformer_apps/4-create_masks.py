#!/usr/bin/env python3
""" Create Masks """

import tensorflow as tf


def create_masks(inputs, target):
    """ Function that creates all masks for training/validation """

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    size = target.shape[1]
    lookAheadMask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    dec_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_mask = dec_target_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(dec_target_mask, lookAheadMask)

    return encoder_mask, combined_mask, decoder_mask
