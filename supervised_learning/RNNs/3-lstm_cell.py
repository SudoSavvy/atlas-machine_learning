#!/usr/bin/env python3
"""
Class that represents an LSTM (Long Short-Term Memory) cell
"""

import numpy as np


class LSTMCell:
    """
    Represents a single LSTM cell performing one step of
    forward propagation in an LSTM-based RNN.
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data input
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs

        Public instance attributes:
            Wf, bf: forget gate weights and bias (shape: (i + h, h), (1, h))
            Wu, bu: update gate weights and bias (shape: (i + h, h), (1, h))
            Wc, bc: candidate cell state weights and bias (shape: (i + h, h), (1, h))
            Wo, bo: output gate weights and bias (shape: (i + h, h), (1, h))
            Wy, by: output weights and bias (shape: (h, o), (1, o))
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
            c_prev (np.ndarray): previous cell state, shape (m, h)
            x_t (np.ndarray): input data at time t, shape (m, i)

        Returns:
            h_next (np.ndarray): next hidden state, shape (m, h)
            c_next (np.ndarray): next cell state, shape (m, h)
            y (np.ndarray): output of the cell, shape (m, o)
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)

        # Update/input gate
        u_t = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)

        # Candidate cell state
        c_hat = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_hat

        # Output gate
        o_t = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
