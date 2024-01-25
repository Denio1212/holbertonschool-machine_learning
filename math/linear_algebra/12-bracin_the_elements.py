#!/usr/bin/env python3
"""
Performs element-wise addition, multiplication, subtraction, and division
"""
import numpy as np


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, multiplication, subtraction, and division
    Args:
        mat1: first element
        mat2: second element

    Returns: the result of the addition, multiplication, and division
    """
    return (np.add(mat1, mat2), np.multiply(mat1, mat2),
            np.subtract(mat1, mat2), np.divide(mat1, mat2))
