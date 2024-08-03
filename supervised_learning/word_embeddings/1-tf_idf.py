#!/usr/bin/env python3
"""
Function to create tf-idf embedding matrix
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Function to create tf-idf embedding matrix

    Params:
        sentences (list): list of sentences to analyze
        vocab (list): list of vocabulary words to be used
            -> if None use the entire sentences list

    Returns:
        embeddings (np array): with shape (s, f) containing the embeddings
            s -> number of sentences in "sentences"
            f -> number of features to be analyzed
        features (list): list of feature names used for embeddings
    """
    if vocab is None:
        vector = TfidfVectorizer()
    else:
        vector = TfidfVectorizer(vocabulary=vocab)

    tfid_matrix = vector.fit_transform(sentences)
    embeddings = tfid_matrix.toarray()
    feature_names = vector.get_feature_names_out()

    return embeddings, feature_names
