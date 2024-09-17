#!/usr/bin/env python3
"""
Initializes the cluster Centroids for k-means clustering.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes the cluster Centroids for k-means clustering.

    Parameters:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        -> n is the number of data points in the dataset
        -> d is the number of dimensions in the dataset

        k is a positive integer containing the number of clusters
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k < 1:
        return None

    n, d = X.shape
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    return np.random.uniform(x_min, x_max, size=(k, d))
