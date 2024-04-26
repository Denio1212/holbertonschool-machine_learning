#!/usr/bin/env python3
"""
Calculates the f1 score// 2 * (precision * recall / (precision + recall)) //
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the f1 score// 2 * (precision * recall / (precision + recall))

    :param confusion: confusion matrix of shape (classes, classes)
    Classes1 has true labels
    Classes2 has predicted labels

    :return: f1 score // 2 * (precision * recall / (precision + recall))
    """
    num_classes = confusion.shape[0]
    f_1 = np.zeros((num_classes,))

    precisions = precision(confusion)
    sensitivites = sensitivity(confusion)

    for i in range(num_classes):
        f_1[i] = ((2 * (precisions[i] * sensitivites[i]))
                  / (precisions[i] + sensitivites[i]))

    return f_1
