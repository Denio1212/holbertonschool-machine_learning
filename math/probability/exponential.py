#!/usr/bin/env python3
"""
A class named exponential that represents an exponential distribution
"""


class Exponential:
    """
    represents an exponential distribution
    has a class constructor with:
    data: a list of data to be used to estimate exponential distribution
    lambtha: expected number of times the exponential distribution occurs
    """
    def __init__(self, data=None, lambtha=1.):
        """
        represents an exponential distribution
        :param data: a list of data to be used to estimate exponential
        :param lambtha: expected number of times the exponential distribution
        Sets the instance variables lambtha, as a float

        if the data is not given, i.e None, use the given lambtha value

        if lambtha is Not positive, raise a value error:
        "lambtha must be a positive value"/

        if data is given, calculate the lambtha of that data

        if data is not a list, raise type error:
        "data must be a list"

        if data does not contain at least two elements, value error:
        "data must contain multiple values"
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 1:
                raise ValueError("data must contain multiple values")

            self.lambtha = 1 / sum(data) / len(data)
