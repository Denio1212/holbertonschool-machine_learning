#!/usr/bin/env python3
"""
The Neural Network Deepens
"""

import numpy as np


class DeepNeuralNetwork:
    """
    The Deep Neural Network
    """

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):
            if not isinstance(layer, int) or layer < 1:
                raise TypeError('layers must be a list of positive integers')

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (np.random.randn(layer, previous) *
                                            np.sqrt(2 / previous))
            previous = layer

        self.L = len(layers)
        self.cache = {}
        self.weights = weights
