#!/usr/bin/env python3
""" Shear Image """
import tensorflow as tf


def shear_image(image, intensity):
    """ Function that randomly shears an image """
    arrayImage = tf.keras.preprocessing.image.img_to_array(image)
    shearedImage = tf.keras.preprocessing.image.random_shear(arrayImage,
                                                             intensity,
                                                             row_axis=0,
                                                             col_axis=1,
                                                             channel_axis=2
                                                             )
    imageOut = tf.keras.preprocessing.image.array_to_img(shearedImage)
    return imageOut
