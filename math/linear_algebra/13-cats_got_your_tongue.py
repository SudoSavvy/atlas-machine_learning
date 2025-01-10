#!/usr/bin/env python3
"""
Module for concatenating two matrices along a specific axis.

This module provides a function to concatenate two matrices (numpy.ndarrays) 
along a specific axis using numpy's concatenate function. 

Example:
    mat1 = np.array([[1, 2, 3], [4, 5, 6]])
    mat2 = np.array([[7, 8, 9], [10, 11, 12]])
    result = np_cat(mat1, mat2, axis=0)
    print(result)  # Output: Concatenated matrix along axis 0
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.
        axis (int): The axis along which the matrices will be concatenated. Default is 0.

    Returns:
        numpy.ndarray: The concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
