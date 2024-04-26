#!/usr/bin/env python3
"""
Makes a Confusion Matrix in numpy only
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    :labels: list of correct labels

    :logits: list of predicted labels

    :return: confusion matrix

    For the construction of the confusion matrix you need:

    The unique values of the labels list
    The number of classes in the labels list
    And a predefined confusion matrix (With empty values for now)

    You use the unique values of the labels list to find the unique values,
    Which we use to get the number of classes in the labels list,
    Which we use to make the empty confusion matrix

    Afterwards we loop over the true labels and define the true and
    predicted labels
    After finding them we add them to the confusion matrix
    """
    unique_classes = labels.shape[0]
    num_classes = labels.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(unique_classes):
        true_class = np.argmax(labels[i])
        predicted_class = np.argmax(logits[i])

        confusion_matrix[true_class, predicted_class] += 1

    return confusion_matrix
