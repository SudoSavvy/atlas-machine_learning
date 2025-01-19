#!/usr/bin/env python3

class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.

        Args:
            data (list): List of data points (optional).
            mean (float): Mean of the distribution.
            stddev (float): Standard deviation of the distribution.

        Raises:
            ValueError: If stddev is not a positive value or equals 0.
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate mean
            self.mean = sum(data) / len(data)
            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value corresponding to z.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        # Calculate the PDF using the formula:
        pi = 3.141592653589793
        sqrt_2pi = (2 * pi) ** 0.5

        coeff = 1 / (self.stddev * sqrt_2pi)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)

        # Calculate exp(exponent) manually for precision
        exp_value = 1
        term = 1
        n = 1
        while True:
            term *= exponent / n
            exp_value += term
            n += 1
            if abs(term) < 1e-15:
                break

        return coeff * exp_value
