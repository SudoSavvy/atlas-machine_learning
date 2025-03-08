#!/usr/bin/env python3

import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer of a neural network.
    
    Parameters:
    - dA (numpy.ndarray): Partial derivatives with respect to the output of the pooling layer (m, h_new, w_new, c_new)
    - A_prev (numpy.ndarray): Output of the previous layer (m, h_prev, w_prev, c)
    - kernel_shape (tuple): Size of the kernel for the pooling (kh, kw)
    - stride (tuple): Strides for the pooling (sh, sw)
    - mode (str): Either 'max' or 'avg', indicating the type of pooling
    
    Returns:
    - dA_prev (numpy.ndarray): Partial derivatives with respect to the previous layer
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, _ = dA.shape
    
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(h_new):
        for j in range(w_new):
            h_start, h_end = i * sh, i * sh + kh
            w_start, w_end = j * sw, j * sw + kw
            
            for n in range(m):
                for ch in range(c):
                    if mode == 'max':
                        a_slice = A_prev[n, h_start:h_end, w_start:w_end, ch]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[n, h_start:h_end, w_start:w_end, ch] += mask * dA[n, i, j, ch]
                    elif mode == 'avg':
                        avg_value = dA[n, i, j, ch] / (kh * kw)
                        dA_prev[n, h_start:h_end, w_start:w_end, ch] += avg_value
    
    return dA_prev
