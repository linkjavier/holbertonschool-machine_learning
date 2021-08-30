#!/usr/bin/env python3
""" Hue Image """
import tensorflow as tf


def change_hue(image, delta):
    """ Function that changes the hue of an image"""
    return tf.image.adjust_hue(image, delta)
