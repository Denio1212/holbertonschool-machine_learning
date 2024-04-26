#!/usr/bin/env python3
"""
Calculates the specificity of all classes in confusion matrix
"""


import numpy as np


def specificity(confusion):
    """
    Calculates the specificity of all classes in confusion matrix

    :param confusion: confusion matrix shape (classes, classes)
    Classes1 carries true values
    Classes2 carries predicted values

    :return: specificity
    """
    num_classes = confusion.shape[0]
    specificity = np.zeros((num_classes,))

    for i in range(num_classes):
        t_posititve = confusion[i][i]

        f_positive = np.sum(confusion[:, i]) - t_posititve

        f_negative = np.sum(confusion[i, :]) - t_posititve

        t_negative = np.sum(confusion) - (t_posititve - f_positive, f_negative)

        specificity[i] = t_negative / (t_negative + f_positive)

    return specificity
