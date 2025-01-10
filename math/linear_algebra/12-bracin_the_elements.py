#!/usr/bin/env python3
def np_elementwise(mat1, mat2):
    # Perform element-wise operations manually without using numpy
    add = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    sub = [[mat1[i][j] - mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    mul = [[mat1[i][j] * mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    div = [[mat1[i][j] / mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    
    # Return the results as a tuple
    return add, sub, mul, div
