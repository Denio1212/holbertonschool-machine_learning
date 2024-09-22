#!/usr/bin/env python3
"""
Calculates a GMM from a dataset
"""


import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
            -> n: number of data points
            -> d: dimensionality of the dataset

        k: number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) containing
        the cluster priors

        m: numpy.ndarray of shape (k, d) containing
        the centroid

        S: a numpy.ndarray of shape (k, d, d)
        containing the covariance matrices

        clss: a numpy.ndarray of shape (n,) containing
        the cluster indices for each data point

        bic: is a numpy.ndarray of shape (kmax - kmin + 1)
        containing the BIC value for each cluster size tested
    """
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    gmm_model.fit(X)

    pi = gmm_model.weights_

    m = gmm_model.means_

    S = gmm_model.covariances_

    clss = gmm_model.predict(X)

    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
