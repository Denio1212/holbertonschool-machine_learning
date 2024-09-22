#!/usr/bin/env python3
"""
Perfroms k-means on a dataset using sklearn
"""


import sklearn.cluster


def kmeans(X, k):
    """
    Performs k-means on a dataset using sklearn

    Parameters:
        X: numpy array shape: (n, d)
            -> n: number of datapoints
            -> d: number of dimensions

        k: number of clusters

    Returns:
        C: numpy array shape: (k, d) containing the centroids
        means for each clusters

        clss: numpy arrar shaped (n,) containing the
        index of the cluster in C that each datapoint belongs to
    """
    Kmeans = sklearn.cluster.KMeans(n_clusters=k)

    Kmeans.fit(X)

    C = Kmeans.cluster_centers_

    clss = Kmeans.labels_

    return C, Kmeans.labels_
