#!/usr/bin/env python3

def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    on two matrices (or a matrix and scalar) without using loops or conditionals.

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

    # Element-wise operations using NumPy (without loops or conditionals)
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
