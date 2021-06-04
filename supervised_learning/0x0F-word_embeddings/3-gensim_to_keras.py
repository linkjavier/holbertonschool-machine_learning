#!/usr/bin/env python3
""" Extract Word2Vec """

from gensim.models import Word2Vec


def gensim_to_keras(model):
    """ Function that converts a gensim word2vec
        model to a keras Embedding layer
    """
    layer = model.wv.get_keras_embedding()

    return layer
