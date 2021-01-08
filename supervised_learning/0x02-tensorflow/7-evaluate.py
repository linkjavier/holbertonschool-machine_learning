#!/usr/bin/env python3
""" Tensorflow module """

import tensorflow as tf


def evaluate(X, Y, save_path):
    """ Function that evaluates the output of a neural network. """

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(sess, '{}'.format(save_path))

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[]

        predictionOutput = sess.run(y_pred, feed_dict={x: X, y: Y})
        lossOutput = sess.run(loss, feed_dict={x: X, y: Y})
        accuracyOutput = sess.run(accuracy, feed_dict={x: X, y: Y})

    return predictionOutput, lossOutput, accuracyOutput
