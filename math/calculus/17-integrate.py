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
    if not isinstance(C, int):
        return None
    if not poly:
        return None
    integral_result = [coef / (index + 1) for index, coef in enumerate(poly)]
    integral_result.insert(0, C)
    integral_result = [int(coef) if int(coef) == coef
                       else coef for coef in integral_result]
    while integral_result[-1] == 0 and len(integral_result) > 1:
        integral_result.pop()
    return integral_result
