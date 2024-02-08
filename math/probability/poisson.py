#!/usr/bin/env python3
"""
represent a poisson distribution
"""


class Poisson:
    def __init__(self, data=None, lambtha=1):
        """
        represents a poisson distribution
        :param self: refers to the class instance
        :param data: the data to be used for the poisson distribution
        :param lambtha: the value of the lamda parameter
        :return: A poisson distribution
        """
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        :param self: refers to the class instance
        :param k: value to be used for the poisson distribution
        :return: the value of the PMF distribution
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        pmf = ((e ** -lambtha) * lambtha ** k) / factorial
        return pmf
