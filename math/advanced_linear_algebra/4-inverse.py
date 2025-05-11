#!/usr/bin/env python3
"""
This module defines a function to compute the inverse of a square matrix.
"""


def determinant(matrix):
    """
    Computes the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix.

    Returns:
        int or float: The determinant.
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
        matrix (list of lists): The matrix.

    Returns:
        list of lists: Cofactor matrix.
    """
    n = len(matrix)
    if n == 1:
        return [[1]]

    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r != i:
                    sub_row = []
                    for c in range(n):
                        if c != j:
                            sub_row.append(matrix[r][c])
                    sub.append(sub_row)
            row.append(((-1) ** (i + j)) * determinant(sub))
        cof.append(row)
    return cof


def adjugate(matrix):
    """
    Computes the adjugate (adjoint) matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix.

    Returns:
        list of lists: Adjugate matrix.
    """
    cof = cofactor(matrix)
    n = len(matrix)
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cof[j][i])
        adj.append(row)
    return adj


def inverse(matrix):
    """
    Computes the inverse of a square matrix.

    Args:
        matrix (list of lists): The matrix to invert.

    Returns:
        list of lists or None: The inverse matrix, or None if singular.

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

    det = determinant(matrix)
    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    adj = adjugate(matrix)
    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inv.append(row)

    return inv
