#!/usr/bin/env python3
"""
This module performs matrix multiplication for two matrices.

It contains a function `mat_mul` which takes two 2D matrices
(lists of lists),
multiplies them if they are compatible (i.e., the number of columns
in the first
matrix is equal to the number of rows in the second), and returns
the resulting matrix.

If the matrices are incompatible, the function returns None.

Example:
    mat1 = [[1, 2], [3, 4], [5, 6]]
    mat2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))  # Expected Output: [[11, 14, 17, 20]
      [23, 30, 37, 44], [35, 46, 57, 68]]
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices.

    Args:
        mat1 (list of lists): The first matrix represented as a 2D list.
        mat2 (list of lists): The second matrix represented as a 2D list.

    Returns:
        list of lists: The result of the matrix multiplication
          or None if the matrices
                       cannot be multiplied due to incompatible dimensions.
    """

# Check the number of columns in mat1 matches the number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Perform matrix multiplication
    result = [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
               for j in range(len(mat2[0]))] for i in range(len(mat1))]
    return result


if __name__ == "__main__":
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]

    print(mat_mul(mat1, mat2))
    # Expected Output: [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
