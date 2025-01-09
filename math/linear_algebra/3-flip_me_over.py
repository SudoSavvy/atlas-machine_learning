#!/usr/bin/env python3

def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    print(mat1)
    print(matrix_transpose(mat1))  # Output: [[1, 3], [2, 4]]

    mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    print(mat2)
    print(matrix_transpose(mat2))  # Output: [[1, 6, 11, 16, 21, 26], [2, 7, 12, 17, 22, 27], [3, 8, 13, 18, 23, 28], [4, 9, 14, 19, 24, 29], [5, 10, 15, 20, 25, 30]]
