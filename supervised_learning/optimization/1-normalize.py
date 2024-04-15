#!/usr/bin/env python3
"""
Normalizes a matrix
"""


import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix

    :param X: matrix to be normalized with shape (d, nx)
    d is the number of data points
    nx number of features

    :param m: mean of all features with shape (nx,)

    :param s: standard deviation of all features with shape (nx,)

    :return: normalized matrix
    """
    return (X - m) / s