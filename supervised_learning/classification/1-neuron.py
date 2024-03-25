#!/usr/bin/env python3
"""
Adds getters and makes the instances private
"""


import numpy as np


class Neuron:
    """
    Neuron class
    """
    def __init__(self, nx):
        """
        Initializer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def A(self):
        return self.__A

    @property
    def b(self):
        return self.__b
