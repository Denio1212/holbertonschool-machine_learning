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
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = n
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                variance = (summation / len(data))
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p

    def pmf(self, k):
        """
        calculates the probability mass function of k
        :param k: number of successes
        :return: pmf value for k
        """
        n_factorial = 1
        k_factorial = 1
        nk_factorial = 1
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        for i in range(self.n):
            n_factorial *= (i + 1)
        for j in range(k):
            k_factorial *= (j + 1)
        for ij in range(self.n - k):
            nk_factorial *= (ij + 1)
        nk_binom = n_factorial / (k_factorial * nk_factorial)
        return nk_binom * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of cdf for a given k value
        :param k: number of successes
        :return: cdf value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
