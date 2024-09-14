#!/usr/bin/env python3
"""
Performs PCA on a given dataset
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a given dataset
    """
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return T
