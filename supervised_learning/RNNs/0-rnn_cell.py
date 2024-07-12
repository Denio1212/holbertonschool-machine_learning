#!usr/bin/env python3
"""
Makes a simple RNN cell
"""

import numpy as np


class RNNCell:
    """
    RNN cell class
    """

    def __init__(self, i, h, o):
        """
        RNN cell constructor

        :param i: dimension of data

        :param h: dimension of hidden state

        :param o: dimension of output state

        Creates public instance attributes:
        Wh, Wy, bh, by

        (Wh, Wy) are for the concatenated hidden state and input data
        (bh, by) are for the output

        The weights will be initialized using random normal initialization

        The biases will be 0
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward pass of the RNN cell in one step

        :param x_t: numpy array of shape (m, i) that contains input
        for the cell

        :param h_prev: numpy array of shape (m, h) that contains hidden
        data

        -> m is the batch size of the data
        """

        def softmax(x):
            """
            computes the softmax activation function

            :param x: input
            """
            return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

        h_x = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(h_x, self.Wh) + self.bh)

        output_tanh = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, output_tanh
