#!/usr/bin/env python3
"""
Multiplies two matrices using linear algebra
"""


def mat_mul(mat1, mat2):
    """
    Args:
        mat1: the first matrix
        mat2: the second matrix

    Returns: Matrix with the multiplications of two matrices
    """
    mat_multipla = []
    if len(mat1[0]) != len(mat2):
        return None
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            element = 0
            for k in range(len(mat2)):
                element += mat1[i][k] * mat2[k][j]
            row.append(element)
        mat_multipla.append(row)
    return mat_multipla
