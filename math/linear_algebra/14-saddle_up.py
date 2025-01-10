#!/usr/bin/env python3
"""
Module that contains a function for performing matrix multiplication.

This module provides a single function, np_matmul, which takes two matrices
as inputs and returns their matrix product using NumPy's matmul function.

Example:
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    result = np_matmul(mat1, mat2)
    print(result)  # Output: [[19 22] [43 50]]
"""

import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrix multiplication.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of the matrix multiplication.
    """
    return np.matmul(mat1, mat2)
