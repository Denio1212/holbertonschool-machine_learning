#!/usr/bin/env python3
"""
Function that returns the transpose of a given matrix by swapping the
rows and columns
"""


def matrix_transpose(matrix):
    """
    Args:
        matrix:

    Returns:
    A new matrix with the transpose of the given matrix
    """
    matrix_rows = len(matrix)
    matrix_cols = len(matrix[0]) if matrix else 0

    transposed_matrix = [[matrix[j][i] for j in range(matrix_rows)]
                         for i in range(matrix_cols)]
    return transposed_matrix
