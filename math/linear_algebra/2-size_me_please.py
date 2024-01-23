#!/usr/bin/env python3
"""
The following code shows how to access the matrix and get its shape
Originally I didn't have the isinstance but since the matrix wasn't being
recognised as a list i put it in.
"""


def matrix_shape(matrix):
    """Gets the shape of the matrix"""
    shape = []
    while matrix and isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
