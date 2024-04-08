#!/usr/bin/env python3
"""
Converts a vector of integers into a one-hot matrix
"""


import tensorflow.keras as keras


def one_hot(labels, classes=None):
    """
    Converts a matrix of integers into a one-hot matrix
    """
    return keras.utils.to_categorical(labels, num_classes=classes)

