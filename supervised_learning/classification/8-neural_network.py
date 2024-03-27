#!/usr/bin/env python3
"""
A neural network with one hidden layer performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    A neural network with one hidden layer performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        nx is the number of input features

        nodes is the number of nodes in the hidden layer
        """
        self.nodes = nodes
        self.nx = nx
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.randn(nx, nodes)
        self.W2 = np.random.randn(1, nodes)
        self.b1 = np.zeros((1, nodes))
        self.b2 = 0
        self.A1 = 0
        self.A2 = 0
