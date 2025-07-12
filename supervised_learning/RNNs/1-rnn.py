#!/usr/bin/env python3
"""
Function that performs forward propagation for a simple RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Args:
        rnn_cell: instance of RNNCell used for forward propagation
        X (np.ndarray): input data of shape (t, m, i)
                        t: number of time steps
                        m: batch size
                        i: dimensionality of the data
        h_0 (np.ndarray): initial hidden state of shape (m, h)
                          h: dimensionality of the hidden state

    Returns:
        H (np.ndarray): array containing all hidden states, shape (t + 1, m, h)
        Y (np.ndarray): array containing all outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for time_step in range(t):
        h_prev = H[time_step]
        x_t = X[time_step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[time_step + 1] = h_next
        Y[time_step] = y

    return H, Y
