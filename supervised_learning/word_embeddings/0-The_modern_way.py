#!/usr/bin/env python3
"""
Creates a Bag of Words embedding matrix
"""


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a Bag of Words embedding matrix

    Parameters:
        sentences (list): list of sentences for analysis
        vocab (list): list of vocabulary words to use for analysis
            - If none all words within sentences are used

    Returns:
        embeddings (numpy.ndarray): shape (s, f) containing the embeddings
            - s -> number of sentences in "sentences"
            - f -> number of features analyzed
        features (list): list of features used for embeddings
    """
    if vocab is None:
        vector = CountVectorizer()
    else:
        vector = CountVectorizer(vocabulary=vocab)

    pre_output = vector.fit_transform(sentences)
    feature_names = vector.get_feature_names_out()

    return pre_output, feature_names
