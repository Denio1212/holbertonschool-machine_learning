#!/usr/bin/env python3
"""
Returns the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Returns the derivative of a polynomial
    :param poly: the given polynomial
    :return: The derivative of the polynomial
    """
    if not isinstance(poly, (list, tuple)):
        return None
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
    if not derivative:
        derivative = [0]
    return derivative
