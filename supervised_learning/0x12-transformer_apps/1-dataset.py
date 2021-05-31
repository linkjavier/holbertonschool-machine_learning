#!/usr/bin/env python3
""" Encode Tokens """

import tensorflow_datasets as tfds


class Dataset():
    """ Loads and preps a dataset for machine translation"""

    def __init__(self):
        """ Constructor """

        ds, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                          with_info=True, as_supervised=True)
        self.data_train = ds['train']
        self.data_valid = ds['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

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
