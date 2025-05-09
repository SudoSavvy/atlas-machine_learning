#!/usr/bin/env python3
"""
This module defines a function to compute the determinant of a matrix.
"""


def determinant(matrix):
    """
    Computes the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix to compute the determinant of.

    Returns:
        int or float: The determinant value.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.
    """
    # Validate input type
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix (i.e., [[]])
    if matrix == [[]]:
        return 1

    # Check square shape
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a square matrix")

    # Base cases
    if size == 0:
        return 1
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case (Laplace expansion)
    det = 0
    for col in range(size):
        # Build minor matrix
        minor = []
        for r in range(1, size):
            row = []
            for c in range(size):
                if c != col:
                    row.append(matrix[r][c])
            minor.append(row)
        # Compute cofactor and recurse
        cofactor = (-1) ** col * matrix[0][col] * determinant(minor)
        det += cofactor

    return det
