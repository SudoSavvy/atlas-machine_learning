#!/usr/bin/env python3

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters:
    - A_prev (numpy.ndarray): Input data of shape (m, h_prev, w_prev, c_prev)
    - kernel_shape (tuple): Size of the kernel (kh, kw)
    - stride (tuple): Stride of the pooling operation (sh, sw)
    - mode (str): Pooling mode ('max' or 'avg')

    Returns:
    - numpy.ndarray: Output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A_pool = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            h_start, h_end = i * sh, i * sh + kh
            w_start, w_end = j * sw, j * sw + kw

            slice_A = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A_pool[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A_pool[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return A_pool
