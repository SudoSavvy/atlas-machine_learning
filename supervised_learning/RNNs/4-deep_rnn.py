#!/usr/bin/env python3
"""
Function that performs forward propagation for a deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Args:
        rnn_cells (list): list of RNNCell instances (length l, number of layers)
        X (np.ndarray): input data for the RNN, shape (t, m, i)
                        t: number of time steps
                        m: batch size
                        i: dimensionality of the data
        h_0 (np.ndarray): initial hidden states, shape (l, m, h)
                          l: number of layers
                          h: dimensionality of hidden state

    Returns:
        H (np.ndarray): all hidden states, shape (t + 1, l, m, h)
        Y (np.ndarray): all outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]  # Output dimension from last cell

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for time_step in range(t):
        x = X[time_step]

        for layer in range(l):
            h_prev = H[time_step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[time_step + 1, layer] = h_next
            x = h_next  # Pass to next layer

        Y[time_step] = y  # Only the last layer's output is used

    return H, Y
