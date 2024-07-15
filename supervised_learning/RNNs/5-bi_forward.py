#!/usr/bin/env python3
"""
Defines the class BidirectionalCell that represents a bidirectional RNN cell
"""


import numpy as np


class BidirectionalCell:
    """
    Represents a birectional RNN cell

    """
    def __init__(self, i, h, o):
        """
        Class constructor

        """
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next
