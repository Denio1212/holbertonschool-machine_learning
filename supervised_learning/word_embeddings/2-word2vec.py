#!/usr/bin/env python3
""""
Creates and trains a gensim word2vec model
"""


from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model

    Parameters:
        sentences (list): list of sentences to be trained on

        size (int): dimensionality of embedding layer

        min_count (int): minimum count of words to be considered in training

        window (int): maximum difference between the current predicted word
        and the sentence

        negative (int): negative sampling

        cbow (boolean): if true, use cbow embeddings, if false, use skip-gram

        iterations (int): number of training iterations

        seed (int): random seed for number generator

        workers (int): number of parallel worker threads to train the model
    """
    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1

    model = Word2Vec(
        sentences,
        size,
        min_count,
        window,
        negative,
        sg=cbow_flag,
        iter=iterations,
        seed=seed,
        workers=workers
    )

    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model

