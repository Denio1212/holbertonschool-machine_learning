#!/usr/bin/env python3
"""
Calculates the sensitivity of each class
"""


import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity of each class

    :param confusion: confusion array of shape (classes, classes), where,
    classes -> number of classes

    :return: sensitivity of each class in a numpy array
    """
    num_classes = confusion.shape[0]
    sensitivity = np.zeros((num_classes,))

    for i in range(num_classes):
        true_positives = confusion[i, i]

        true_total = np.sum(confusion[i, :])

        sensitivity[i] = true_positives / true_total

    return sensitivity
