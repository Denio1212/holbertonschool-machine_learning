#!/usr/bin/env python3
"""
Creates a class that represents normal distribution
"""


class Normal:
    """
    Houses a constructor with a mean and a standard deviation
    """
    def __init__(self, data=None, mean=0, stddev=1.):
        """
        :param data: the list of data used
        :param mean: mean of the deviation
        :param stddev: standard deviation of the distribution
        sets the instance attributes mean and stddev as floats

        if data is not given, use the given mean and standard deviation

        if stddev is not a positive value or is 0 raise value error:
        "stddev must be a positive value"

        if data is given, calculate the mean and standard deviation of data
        if data is not a list, raise type error:
        "data must be a list"

        if data is < 2, raise value error:
        "data must contain multiple values"
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum((ent - self.mean) ** 2 for ent in data)
                          / len(data)) ** 0.5

    def z_score(self, x):
        """
        Calculates the z score of a given value
        :param x: the given x value
        :return: z_score of x
        """
        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
        calculates the x value of a given z-score
        :param z: the z-score of x
        :return: x-value of z
        """
        return float(self.mean + z * self.stddev)
