#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    # Check if concatenation along the specified axis is possible
    if axis == 0:
        # Ensure both matrices have the same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Concatenate rows
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Ensure both matrices have the same number of rows
        if len(mat1) != len(mat2):
            return None
        # Concatenate columns row by row
        return [mat1[i] + mat2[i] for i in range(len(mat1))]


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]

    mat4 = cat_matrices2D(mat1, mat2)
    mat5 = cat_matrices2D(mat1, mat3, axis=1)

    print(mat4)  # Expected: [[1, 2], [3, 4], [5, 6]]
    print(mat5)  # Expected: [[1, 2, 7], [3, 4, 8]]

    # Show original matrices are unmodified
    mat1[0] = [9, 10]
    mat1[1].append(5)
    print(mat1)  # Expected: [[9, 10], [3, 4, 5]]
    print(mat4)  # Expected: [[1, 2], [3, 4], [5, 6]]
    print(mat5)  # Expected: [[1, 2, 7], [3, 4, 8]]
