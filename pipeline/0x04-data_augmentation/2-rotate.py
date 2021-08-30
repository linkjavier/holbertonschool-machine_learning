#!/usr/bin/env python3
""" Rotate Image """
import tensorflow as tf


def rotate_image(image):
    """ Function that rotates an image by 90 degrees counter-clockwise """
    return tf.image.rot90(image, k=1)
