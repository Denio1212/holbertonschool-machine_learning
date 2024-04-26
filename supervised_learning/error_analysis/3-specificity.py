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

    for i in range(num_classes):
        true_negatives = (np.sum(confusion) - np.sum(confusion, axis=0) -
                          np.sum(confusion, axis=1) + np.diag(confusion))

        false_positives = np.sum(confusion, axis=0) - np.diag(confusion)

        specificity_per_class = true_negatives / (true_negatives +
                                                  false_positives)

        return specificity_per_class
