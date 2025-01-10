#!/usr/bin/env python3
import numpy as np
matrix = np.array([[36, 14, 57, 82, -9, 10], [100, 109, -36, 7, 2, 443], [6, 54, 67, 3, 3, 1], 
                   [57, 82, 23, 72, 45, 60], [23, 72, 12, 21, 54, 44], [12, 5, 44, 30, 11, 67]])
mat1 = matrix[1:3]; mat2 = matrix[:, 2:4]; mat3 = matrix[3:, 3:]
print(f"The middle two rows of the matrix are:\n{mat1}")
print(f"The middle two columns of the matrix are:\n{mat2}")
print(f"The bottom-right, square, 3x3 matrix is:\n{mat3}")
