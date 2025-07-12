#!/usr/bin/env python3
"""
Class that represents a bidirectional RNN cell with forward and backward methods
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data input
            h (int): dimensionality of the hidden states
            o (int): dimensionality of the outputs

        Public instance attributes:
            Whf (np.ndarray): weights for the forward direction, shape (i + h, h)
            bhf (np.ndarray): bias for the forward direction, shape (1, h)
            Whb (np.ndarray): weights for the backward direction, shape (i + h, h)
            bhb (np.ndarray): bias for the backward direction, shape (1, h)
            Wy (np.ndarray): weights for the outputs, shape (2h, o)
            by (np.ndarray): bias for the outputs, shape (1, o)
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step

        Args:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
            x_t (np.ndarray): input data at time t, shape (m, i)

        Returns:
            h_next (np.ndarray): next hidden state, shape (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Args:
            h_next (np.ndarray): next hidden state, shape (m, h)
            x_t (np.ndarray): input data at time t, shape (m, i)

        Returns:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev
