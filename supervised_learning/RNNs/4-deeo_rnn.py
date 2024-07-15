#!/usr/bin/env python3
"""
Performs Forward Propagation in a deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation of a deep RNN

    :param rnn_cells: list of RNNCell instances with length l
    l - number of layers

    :param X: numpy array of shape (t, m, i) that contains input
    t -> max number of time steps
    m -> batch size of the data
    i -> dimension of data

    h_0 -> initial hidden state, as a numpy array of shape (l, m, h)
    l -> number of layers
    m -> batch size of the data
    h -> dimension of hidden state

    Returns: H, Y
    H -> numpy array with all the hidden states
    Y -> numpy array with all the outputs
    """
    layers = len(rnn_cells)
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    for ts in range(t):
        for layer in range(layers):
            if layer == 0:
                h_prev = X[ts]
            h_prev, y = rnn_cells[layer].forward(H[ts, layer], h_prev)
            H[ts + 1, layer, ...] = h_prev
            if layer == layers - 1:
                if ts == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)

    return H, Y
