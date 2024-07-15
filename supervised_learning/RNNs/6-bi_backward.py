#!/usr/bin/env python3
"""
Performs Forward Propagation in a Bidirectional cell
"""

import numpy as np


class BidirectionalCell:
    """
    The Bidirectional RNN cell

    Creates a cell that can use both forward and backward propagation

    Allows the cell to remember the forward and backward state of the RNN cell
    """

    def __init__(self, i, h, o):
        """
        Initializes the cell

        :param i: dimension of data

        :param h: previous hidden state dimension

        :param o: previous output dimension

        Creates the public instance attributes:
        Whf, Whb, Wy, bhf, bhb, by

        (Whf, bhf) -> are for the hidden states in the forward direction

        (Whb, bhb) -> are for the hidden states in the backward direction

        (Wy, by) -> are for the outputs

        The weights should be initialized using a random normal distribution in the order listed above

        The weights will be used on the right side for matrix multiplication

        The biases should be initialized as zeros
        """
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

        self.bhf = np.random.normal(size=(1, h))
        self.bhb = np.random.normal(size=(1, h))
        self.by = np.random.normal(size=(1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the forward propagation in a single time step

        :param h_prev: previous hidden state dimension, shape (m, h)
        m -> batch size of the data
        h -> hidden state

        h_prev: previous hidden state dimension, shape (m, h)

        Returns: h_next: next hidden state dimension

        c_t -> concatenating the hidden states of the previous time step

        h_next -> next hidden state
        """
        c_t = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(c_t, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Does the same as forward but other way

        :param h_next: next hidden state dimension, shape (m, h)

        :param x_t: input data dimension, shape (m, i)
        m -> batch size of the data

        Returns: h_pev, previous hidden state
        """
        x_b = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(np.matmul(x_b, self.Whb) + self.bhb)

        return h_prev
