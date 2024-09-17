#!/usr/bin/env python3
"""
Calculates the variance of the cluster centroids
"""


import numpy as np


def variance(X, C):
    """
    Calculates the variance of the cluster centroids.
    """
    n, d = X.shape
    centeroids_extend = C[:, np.newaxis]
    distances = np.sqrt(((X - centeroids_extend) ** 2).sum(axis=2))

    min_distances = np.min(distances, axis=0)
    variance = np.sum(min_distances ** 2)

    return variance