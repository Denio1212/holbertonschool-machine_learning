#!/usr/bin/env python3
"""
Creates and trains a simple gensim fasttext model
"""


import gensim


def fasttext_model(
        sentences, size=100, min_count=5, negative=5, window=5, cbow=True,
        iterations=5, seed=0, workers=1
):
    """
    Creates and trains a simple gensim fasttext model

    Parameters:
        sentences (list): list of sentences to be trained on

        size (int): dimensionality of the embedding layer

        min_count (int): minimum count of occurrences of a word in training

        window (int): maximum distance between two words in a sentence

        negative (int): size of the negative sample

        cbow (bool): Determines the training type
        -> if True - Use Cbow
        -> if False - Use Skip-Gram

        iterations (int): number of training iterations

        seed (int): random number seed

        workers (int): number of parallel worker threads
    """
    fast_model = gensim.models.FastText

    model = fast_model(
        sentences,
        size,
        min_count,
        window,
        negative,
        iter=iterations,
        seed=seed,
        workers=workers,
        sg =not cbow
    )
    return model
