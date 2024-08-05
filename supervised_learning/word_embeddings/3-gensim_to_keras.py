#!/usr/bin/env python3
"""
Converts a gensim word2vec model to a keras model
"""


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras model

    Parameters:
        model: gensim word2vec model which is trained
    Returns:
        Keras Embedding layer
    """
    return model.wv.get_keras_embedding(True)
