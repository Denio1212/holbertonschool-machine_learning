#!/usr/bin/env python3
"""
Makes a class representing the GRUCell
"""

import numpy as np


class GRUCell:
    """
    GRU cell class
    """

    def __init__(self, i, h, o):
        """
        GRU cell constructor

        :param i: dimension of data

        :param h: dimension of hidden state

        :param o: dimension of output state

        Creates public instance attributes:
        Wz, Wr, Wh, Wy, bz, br, bh, by

        (Wz, by) are for the update gate
        (Wr, br) are for the reset gate
        (Wh, bh)  are for the intermediate hidden state
        (Wy, by)  are for the output state

        The weights should be initialized using a random normal
        distribution in the order listed above

        The weights will be used on the right side for matrix multiplication

        The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward pass of the GRU cell in one time step

        :param x_t: numpy array of shape (m, i) that contains input
        m -> batch size of the data

        :param h_prev: numpy array of shape (m, h) that contains
        previous hidden state
        m -> batch size of the data

        :return: h_next, y
        h_next -> the next hidden state
        y -> the output of the cell
        """

        def softmax(x):
            """
            computes the softmax activation function
            """
            return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

        h_x = np.concatenate((h_prev, x_t), axis=1)

        r_t = 1 / (1 + np.exp(-(np.dot(h_x, self.Wr) + self.br)))

        z_t = 1 / (1 + np.exp(-(np.dot(h_x, self.Wz) + self.bz)))

        rh_x = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(rh_x, self.Wh) + self.bh)

        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        output = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, output
