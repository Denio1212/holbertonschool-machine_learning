#!/usr/bin/env python3
"""
Converts a numeric label to one-hot matrix
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label to one-hot matrix

    Y is a numpy array of shape (m) where m is the number of examples
    classes is maximum number of classes found in Y

    returns the one-hot encoded numpy array of shape (classes, m) or none
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None

    encode = np.zeros((classes, Y.size), dtype=float)

    encode[Y, np.arange(Y.size)] = 1
    return encode
