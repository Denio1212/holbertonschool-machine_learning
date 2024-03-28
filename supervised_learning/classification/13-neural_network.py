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
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def W2(self):
        return self.__W2

    @property
    def b1(self):
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: array with shape (nx, m) with input data
        nx is the number of input features
        m is the number of examples
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

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
        evaluates the neural network's predictions
        :param X: array with shape (nx, m) with input data
        nx is the number of input features in the neuron
        m is the number of examples
        :param Y: array with shape (1, m) with correct labels for input data
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = np.where(A2 >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        :param X: array with shape (nx, m) with the input data
        nx is the number of input features in the
        m is the number of examples

        :param Y: array with shape (1, m) with correct labels for input data
        :param A1: output of hidden layer
        :param A2: predicted output
        :param alpha: learning rate

        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = Y.shape[1]

        dz2 = (A2 - Y)
        d__W2 = (1 / m) * (np.matmul(dz2, A1.transpose()))
        d__b2 = (1 / m) * (np.sum(dz2, axis=1, keepdims=True))

        dz1 = (np.matmul(self.W2.transpose(), dz2)) * (A1 * (1 - A1))
        d__W1 = (1 / m) * (np.matmul(dz1, X.transpose()))
        d__b1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))

        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)
        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)
