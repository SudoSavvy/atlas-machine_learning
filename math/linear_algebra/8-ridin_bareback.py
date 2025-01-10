#!/usr/bin/env python3


def mat_mul(mat1, mat2):

    # Check if the number of columns in mat1 matches the number of rows in mat2
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
