#!/usr/bin/env python3
"""
Module for performing element-wise matrix operations.

This module provides a function `np_elementwise`
to perform element-wise addition,
subtraction, multiplication, and division on two matrices
(or a matrix and a scalar).
The operations are performed using NumPy without using loops
or conditional statements.

Example:
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    add, sub, mul, div = np_elementwise(mat1, mat2)
    print("Add:", add)
    print("Sub:", sub)
    print("Mul:", mul)
    print("Div:", div)
"""

def np_elementwise(mat1, mat2):


    """
    Perform element-wise addition, subtraction, multiplication,
    and division on two matrices (or a matrix and scalar)
    without using loops or conditionals.

    Args:
        mat1: A 2D matrix (numpy.ndarray).
        mat2: A 2D matrix (numpy.ndarray) or scalar value.

    Returns:
        A tuple containing:
            - element-wise sum
            - element-wise difference
            - element-wise product
            - element-wise quotient
    """

    # Element-wise operations using NumPy
    # (without loops or conditionals)
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
