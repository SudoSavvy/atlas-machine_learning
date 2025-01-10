#!/usr/bin/env python3
import numpy as np

def np_elementwise(mat1, mat2):
    # Perform element-wise operations on the two matrices
    add = np.add(mat1, mat2)
    sub = np.subtract(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)
    
    # Return the results as a tuple
    return add, sub, mul, div
