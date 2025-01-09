#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))  # Output: [[6, 8], [10, 12]]
    print(mat1)  # Output: [[1, 2], [3, 4]]
    print(mat2)  # Output: [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))  # Output: None
