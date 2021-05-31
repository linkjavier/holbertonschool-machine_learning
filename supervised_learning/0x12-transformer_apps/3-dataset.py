#!/usr/bin/env python3
""" Pipeline """

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """ Constructor """

        ds, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                 with_info=True, as_supervised=True)

        self.metadata = metadata
        self.data_train = ds['train']
        self.data_valid = ds['validation']
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def MaxFilter(x, y, max_length=max_len):
            """
            function for .filter() method
            """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        self.data_train = self.data_train.filter(MaxFilter)
        self.data_train = self.data_train.cache()

        train_dataset_size = self.metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(train_dataset_size)
        padded_shapes = ([None], [None])
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=padded_shapes)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)
        self.data_valid = self.data_valid.filter(MaxFilter)
        padded_shapes = ([None], [None])
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=padded_shapes)

    def tokenize_dataset(self, data):
        """ Function that creates sub-word tokenizers for our dataset """

        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ Method that encodes a translation into tokens """

        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """ Method that acts as a tensorflow wrapper
            for the encode instance method
        """

        encode_PT, encode_EN = tf.py_function(func=self.encode, inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])
        encode_PT.set_shape([None])
        encode_EN.set_shape([None])

        return encode_PT, encode_EN
