#!/usr/bin/env python3
"""
Performs PCA on a given dataset
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a given dataset
    """
    covariance_matrix = np.cov(X, rowvar=False)

    eigen_value, eigen_vector = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigen_value)[:: -1]
    sorted_eigen_value = eigen_value[sorted_indices]
    sorted_eigen_vector = eigen_vector[:, sorted_indices]

    cumulative_variance = np.cumsum(sorted_eigen_value) / np.sum(sorted_eigen_value)

    n_componets = np.argmax(cumulative_variance >= var) + 2

    W = sorted_eigen_vector[:, :n_componets]

    return W
