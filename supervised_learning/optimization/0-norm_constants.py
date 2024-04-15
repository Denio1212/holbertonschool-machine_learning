#!/usr/bin/env python3
"""
Calculates the normalization constant for each matrix
"""


import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constant for each matrix

    :param X: matrix to be normalized
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
