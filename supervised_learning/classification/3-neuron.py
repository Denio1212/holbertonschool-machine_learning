#!/usr/bin/env python3
"""
Calculating the cost of the model using logistic regression, Based on 2.neuron
"""

import numpy as np


class Neuron:
    """
    Add the public method def cost(self, Y, A):

    Calculates the cost of the model using logistic regression

    Y is a numpy.ndarray with shape (1, m) that contains the correct
    labels for the input data

    A is a numpy.ndarray with shape (1, m) containing the activated
    output of the neuron for each example
    To avoid division by zero errors, please use 1.0000001 - A

    Returns the cost
    """

    def __init__(self, nx):
        """
        Initializer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A) + (1 - Y) * np.log((1.0000001 - A))))
        costs = (1 / m) * (-m_loss)
        return costs
