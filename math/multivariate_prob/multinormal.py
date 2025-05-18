#!/usr/bin/env python3
"""Module for the Multivariate Normal distribution class."""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Class constructor.

        Parameters:
        - data (numpy.ndarray): shape (d, n) containing the dataset,
                                where d is the number of dimensions
                                and n is the number of data points.

        Sets:
        - mean (numpy.ndarray): shape (d, 1), mean of the dataset
        - cov (numpy.ndarray): shape (d, d), covariance matrix of the dataset

        Raises:
        - TypeError: if data is not a 2D numpy.ndarray
        - ValueError: if data has fewer than 2 data points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Parameters:
        - x (numpy.ndarray): shape (d, 1), data point

        Returns:
        - PDF value at point x

        Raises:
        - TypeError: if x is not a numpy.ndarray
        - ValueError: if x does not have shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        diff = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov)
        exponent = -0.5 * (diff.T @ cov_inv @ diff)

        return float(norm_const * np.exp(exponent))
