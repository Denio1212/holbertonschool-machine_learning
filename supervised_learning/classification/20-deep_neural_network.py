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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: array with shape (nx, m) with input data
        nx is the number of input features
        m is the number of examples
        """
        self.__cache["A0"] = X

        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]

            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            a = 1 / (1 + np.exp(-z))

            self.__cache["A{}".format(index + 1)] = a
        return a, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: array with shape (1, m) with correct labels for input data
        :param A: array with shape (1, m) with activated  outputs
        for each example
        To avoid division by zero errors, we will use
        1.0000001 - A instead of 1 - A
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A) + (1 - Y) * np.log((1.0000001 - A))))
        costs = (1 / m) * (-m_loss)
        return costs

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost
