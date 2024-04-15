#!/usr/bin/env python3
"""
shuffles data points in two matrices the same way.
"""


import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles data points in two matrices the same way.

    Parameters
    X : array-like, shape (m, nx)
    m - number of data points
    n - number of features in X

    Y : array-like, shape (m, ny)
    m - number of data points
    n - number of features in Y

    Returns: Shuffled X, Y
    """
    m = X.shape[0]
    permutate = np.random.permutation(m)

    X_shuffled = X[permutate]
    Y_shuffled = Y[permutate]

    return X_shuffled, Y_shuffled
