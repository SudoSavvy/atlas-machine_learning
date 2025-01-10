#!/usr/bin/env python3
"""
Module for calculating the shape of a NumPy ndarray.

This module provides a function `np_shape` that returns the shape of a given
NumPy matrix as a tuple. The shape describes the number of rows and columns
in a 2D matrix or the dimensions of an N-dimensional array.

Example:
    mat = np.array([[1, 2], [3, 4]])
    shape = np_shape(mat)
    print(shape)  # Output: (2, 2)
"""


def np_shape(matrix):

    """Calculates the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): Input matrix.

    Returns:
        tuple: Shape of the matrix as a tuple of integers.
    """
    return matrix.shape
