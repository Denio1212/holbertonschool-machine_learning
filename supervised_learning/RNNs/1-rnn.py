#!usr/bin/env python3
"""
A cell that performs forward propagation of a simple RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation of a simple RNN

    :param rnn_cell: instance of RNN cell

    :param X: the data to be used, as a numpy array of shape (t, m, i)
    t -> maximum number of time steps
    m -> batch size
    i -> dimension of data

    :param h_0: initial hidden state, as a numpy array of shape (m, h)
    h -> dimension of hidden state
    m -> batch size

    :return: H, Y
    H -> numpy array containing the hidden states
    Y -> numpy array containing the outputs
    """
    H = np.zeros((X.shape[0] + 1, h_0.shape[0], h_0.shape[1]))
    H[0] = h_0

    Y = np.zeros((X.shape[0], X.shape[1], rnn_cell.by.shape[1]))

    for i, x_t in enumerate(X):
        H[i + 1], Y[i] = rnn_cell.forward(H[i], x_t)

    return H, Y
