#!/usr/bin/env python3
"""
Calculates the n-gram BLEU score for a sentence
"""


import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Parameters:
        references (list): list of reference translations
            each reference translation is a list of the words in
            the translation

        sentence (list): list of words in the model proposed sentence

        n: size of the n-gram to be used

    Returns:
        n-gram BLEU score

    Bp is belief propagation
    """
    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))
    n_gram = []
    ngram_references = 0

    for reference in references:
        ngram_references = []
        for i in range(len(reference) - (n - 1)):
            if any(sentence[i:i + n] == reference[j:j + n]
                   for j in range(len(reference) - (n - 1))) and \
                    sentence[i:i + n] not in ngram_references:
                ngram_references.append(sentence[i:i + n])
        n_gram.append(len(ngram_references))

    precision = max(n_gram) / (i + 1)

    return BP * np.exp(np.log(precision))
