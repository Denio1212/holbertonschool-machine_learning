#!/usr/bin/env python3
"""
Performs agglomerative clustering on a dataset
"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset

    Parameters:
        X: numpy array of shape (n, d) containing the dataset
            -> n: number of datapoints
            -> d: dimensionality of the dataset

        dist: maximum cophenetic distance

    Performs agglomerative clustering with a Ward Linkage

    Displays the dendrogram with each cluster displayed in a different colour

    Returns:
        clss: clss, a numpy.ndarray of shape (n,) containing the cluster
        indices for each data point
    """
    ward = scipy.cluster.hierarchy.linkage(X, method='ward')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(ward, color_threshold=dist)
    plt.title('Agglomerative clustering')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(ward, dist, criterion='distance')
    return clss
