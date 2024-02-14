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
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
    if not derivative:
        derivative = [0]
    return derivative
