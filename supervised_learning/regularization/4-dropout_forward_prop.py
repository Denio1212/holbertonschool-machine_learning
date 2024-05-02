#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    :param X: input data with shape (nx, m) containing the input data
        nx -- number of input samples
        m -- number of data points

    :param weights: weights and bias dictionary with shape (w, b)

    :param keep_prob: keep probability for dropout

    :param L: number of hidden layers

    :param keep_prob: keep probability for dropout

    :return: output of the forward propagation in a dictionary
    containing the output of each hidden layer and the dropout mask used on
    each layer

    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    """
    cache = {'A0': X}

    for layer in range(1, L):
        z = (np.matmul(weights["W" + str(layer)],
                       cache["A" + str(layer - 1)]) +
             weights["b" + str(layer)])

        a = np.tanh(z)

        mask = np.random.binomial(1, keep_prob, size=a.shape)
        cache["D" + str(layer)] = mask
        a = np.multiply(a, mask)

        a /= keep_prob
        cache["A" + str(layer)] = a

    z = (np.matmul(weights["W" + str(L)], cache["A" + str(L - 1)]) +
         weights["b" + str(L - 1)])
    a = np.exp(z) / np.sum(np.exp(z), axis=0)

    return cache
