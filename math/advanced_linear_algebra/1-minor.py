#!/usr/bin/env python3
"""
This module defines a function to compute the minor matrix of a square matrix.
"""


def determinant(matrix):
    """
    Computes the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix to compute the determinant of.

    Returns:
        int or float: The determinant value.
    """
    if matrix == [[]]:
        return 1

    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(size):
        sub = []
        for row in range(1, size):
            sub_row = []
            for c in range(size):
                if c != col:
                    sub_row.append(matrix[row][c])
            sub.append(sub_row)
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)

    return det


def minor(matrix):
    """
    Computes the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix to compute the minor matrix of.

    Returns:
        list of lists: The minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    if not isinstance(matrix, list) or any(not isinstance(row, list)
                                           for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r != i:
                    sub_row = []
                    for c in range(n):
                        if c != j:
                            sub_row.append(matrix[r][c])
                    sub.append(sub_row)
            row_minors.append(determinant(sub))
        minors.append(row_minors)

    return minors
