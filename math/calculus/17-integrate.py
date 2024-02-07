#!/usr/bin/env python3
"""
calculates the integral of a polynomial without importing
"""


def poly_integral(poly, C=0):
    """
    :param poly: the given polynomial
    :param C: a constant value
    :return: The integral of the polynomial
    """
    if not isinstance(poly, list):
        return None
    integral_result = [0]
    for power, coef in enumerate(poly):
        if not isinstance(coef, (float, int)):
            return None
        integral_result.append(coef / (power + 1))
        integral_result = [int(coef) if coef.is_integer()
                           else coef for coef in integral_result]
    return integral_result
