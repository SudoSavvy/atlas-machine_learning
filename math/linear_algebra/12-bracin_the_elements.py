#!/usr/bin/env python3

def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    on two matrices (or a matrix and scalar) without using loops or conditionals.

    Args:
        mat1: A 2D matrix (list of lists).
        mat2: A 2D matrix (list of lists) or scalar value.

    Returns:
        A tuple containing:
            - element-wise sum
            - element-wise difference
            - element-wise product
            - element-wise quotient
    """

    # Element-wise addition
    add = [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    
    # Element-wise subtraction
    sub = [[x - y for x, y in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    
    # Element-wise multiplication
    mul = [[x * y for x, y in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    
    # Element-wise division
    div = [[x / y for x, y in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]

    return add, sub, mul, div
