#!/usr/bin/env python3
"""
Performs element-wise addition, multiplication, subtraction, and division
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, multiplication, subtraction, and division
    Args:
        mat1: first element
        mat2: second element

    Returns: the result of the addition, multiplication, and division
    """
    result = []
    result.append(mat1 + mat2)
    result.append(mat1 * mat2)
    result.append(mat1 - mat2)
    result.append(mat1 / mat2)
    return result
