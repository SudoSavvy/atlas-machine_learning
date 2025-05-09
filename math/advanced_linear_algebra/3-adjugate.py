#!/usr/bin/env python3
"""
This module defines a function to compute the adjugate matrix of a square matrix.
"""


def determinant(matrix):
    """
    Computes the determinant of a square matrix.

    Args:
        matrix (list of lists): Matrix to compute the determinant of.

    Returns:
        int or float: Determinant value.
    """
    if matrix == [[]]:
        return 1

    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub = []
        for r in range(1, n):
            row = []
            for c in range(n):
                if c != col:
                    row.append(matrix[r][c])
            sub.append(row)
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)

    return det


def cofactor(matrix):
    """
    Computes the cofactor matrix of a square matrix.

    Args:
        matrix (list of lists): Matrix to compute the cofactor of.

    Returns:
        list of lists: Cofactor matrix.
    """
    n = len(matrix)
    if n == 1:
        return [[1]]

    cof = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r != i:
                    row = []
                    for c in range(n):
                        if c != j:
                            row.append(matrix[r][c])
                    sub.append(row)
            row_cof.append(((-1) ** (i + j)) * determinant(sub))
        cof.append(row_cof)
    return cof


def adjugate(matrix):
    """
    Computes the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): Matrix to compute the adjugate of.

    Returns:
        list of lists: Adjugate matrix.

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

    cof = cofactor(matrix)

    # Transpose the cofactor matrix
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cof[j][i])
        adj.append(row)

    return adj
