#!/usr/bin/env python3
import numpy as np

def np_cat(mat1, mat2, axis=0):

    
    """Concatenates two matrices along a specific axis.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.
        axis (int): The axis along which the matrices will be concatenated.

    Returns:
        numpy.ndarray: The concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)

