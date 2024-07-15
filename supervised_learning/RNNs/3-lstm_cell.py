#!/usr/bin/env python3
"""
Makes a class representing the LSTMCell
"""

import numpy as np


class LSTMCell:
    """
    LSTM cell class

    holds a public instance method named forward and the constructor method
    """

    def __init__(self, i, h, o):
        """
        LSTM cell constructor

        :param i: dimension of data

        :param h: dimension of hidden state

        :param o: dimension of output state

        Creates public instance attributes:
        Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by

        (Wf, bf) -> are for the forget gate
        (Wu, bu) -> are for the update gate
        (Wc, bc) -> are for intermediate cell state
        (Wo, bo) -> are for the output gate
        (Wy, by) -> are for the outputs

        The weights should be initialized using a
        random normal distribution in the order listed above

        The weights will be used on the right side for matrix multiplication

        The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Computes the softmax activation function
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """
        Computes the sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward pass of the LSTM cell in one time step

        :param x_t: numpy array of shape (m, i) that contains input
        m -> batch size of the data

        :param h_prev: numpy array of shape (m, h) that contains
        previous hidden state
        m -> batch size of the data
        h -> the hidden state of the cell

        :param c_prev: numpy array of shape (m, h) that contains
        previous cell state

        The output of the cell should use a softmax activation function

        Returns: h_next, c_next, y
        h_next is the next hidden state

        c_next is the next cell state

        y is the output of the cell

        .T means transposing

        @ means matrix multiplication
        ///

        h_x is the concatenation of the previous hidden state
        and the input

        ft is the forget gate activation

        it is the input/output gate activation

        cct/c_next is the candidate value

        ot is the output gate

        h_next is the next hidden state

        y is the final output of the cell
        """
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        ft = self.sigmoid((h_x.T @ self.Wf) + self.bf)

        it = self.sigmoid((h_x.T @ self.Wu) + self.bu)

        cct = np.tanh((h_x.T @ self.Wc) + self.bc)
        c_next = ft * c_prev + it * cct

        ot = self.sigmoid((h_x.T @ self.Wo) + self.bo)

        h_next = ot * np.tanh(c_next)

        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
