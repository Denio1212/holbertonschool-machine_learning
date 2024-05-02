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
    outputs = {}
    masks = {}

    for layer in range(L - 1):
        z = np.dot(X, weights["W" + str(layer + 1)])

        a = np.tanh(z)

        mask = np.random.randn(a.shape) < keep_prob
        masked_a = a * mask
        outputs["layer_" + str(layer + 1)] = masked_a

        masks["mask_layer_" + str(layer + 1)] = mask

    z = np.dot(outputs["layer" + str(L - 1)], weights["W" + str(L)])
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    outputs["output"] = a

    return outputs, masks
