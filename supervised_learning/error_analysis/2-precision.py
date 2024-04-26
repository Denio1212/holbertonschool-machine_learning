#!/usr/bin/env python3
"""
Calculates the precision of each class.
"""


import numpy as np


def precision(confusion):
    """
    Calculates the precision of each class.

    :param confusion: confusion matrix shape (classes, classes)
    Where classes1 is number of classes with correct predictions.
    And classes2 is number of classes with incorrect predictions.

    :return: precision
    """
    num_classes = confusion.shape[0]
    precision = np.zeros((num_classes,))

    for i in range(num_classes):
        t_positive = confusion[i, i]
        f_positive = np.sum(confusion[:, i]) - t_positive

        precision[i] += t_positive / (t_positive + f_positive)

    return precision
