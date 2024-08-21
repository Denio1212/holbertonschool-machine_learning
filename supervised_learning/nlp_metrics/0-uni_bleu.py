#!/usr/bin/env python3
"""
Calculates the unigram score of BLEU
"""


import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram score of BLEU

    Parameters:
        references (list): list of reference translations
            each reference is a list of words in the translation

        sentence (list): list containing the translation model proposed

    Returns:
        unigram BLEU score
    """
    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))

    precision = max([sum(match in reference for match in set(sentence))
                     for reference in references]) / len(sentence)

    return BP * np.exp(np.log(precision))
