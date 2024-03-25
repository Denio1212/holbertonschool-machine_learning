import numpy as np


class Neuron:
    """
    class that represents a neuron performing binary classification
    """

    def __init__(self, nx):
        """
        nx is the number of input features to the neuron

        if nx is not an integer:
        TypeError: "nx must be an integer"

        if nx is less than 1:
        TypeError: "nx must be a positive integer"

        Public instance attributes:

        W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.

        b: The bias for the neuron. Upon instantiation, it should be initialized to 0.

        A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0