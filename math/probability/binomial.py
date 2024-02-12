#!/usr/bin/env python3
"""
Creates a class named Binomial which represents a binomial distribution
"""


class Binomial:
    """
    class which represents the binomial distribution
    houses a init and the cdf and pdf funcs
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        the init method
        :param data: list of data to be used
        :param n: number of Bernoulli distributions
        :param p: probability of success of each Bernoulli distribution
        sets the instance attributes n and p

        if data is not given:
        we use the given p and n
        if n is not positive:
        Value error -> "n must be positive value"
        if p is not a valid probability:
        Value error -> "p must be greater than 0 and less than 1"

        if data is given:
        calculate n and p from data

        round n to the nearest int -> calc p first then n, then recalculate n

        if data is not list:
        Type Error -> "data must be a list"

        if data is smaller than 2:
        Value error -> "data must contain multiple values"
        """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if self.n <= 0:
                raise ValueError("n must be positive value")
            if 0 >= self.n > 1:
                raise ValueError("p must be greater than 0 and less than 1")
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.p = (2 * (sum(data) / len(data)) / 100)
            self.n = round(len(data) / 2)
