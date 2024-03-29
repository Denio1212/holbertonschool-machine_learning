#!/usr/bin/env python3
"""
Concatenate two 2d matrices..
Checks f#or the axis and if the matrices are the same length.
Afterwards, if the matrix has an axis of 0, it adds them onto the concatenated
matrix.
If the axis is 1, it adds the rows together and then adds the values.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2d matrices
    Args:
        mat1: First 2d matrix
        mat2: Second 2d matrix
        axis: axis (vertical)

    Returns: A concatenated 2d matrix
    """
    concat_2d_mat = []
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        concat_2d_mat = mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        concat_2d_mat = [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
    return concat_2d_mat
