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
        """Calculate the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score."""
        return z * self.stddev + self.mean

    def exp(self, x):
        """Calculate the exponential function for a given x-value."""
        result = 1
        term = 1
        n = 1
        while abs(term) > 1e-15:
            term *= x / n
            result += term
            n += 1
        return result

    def erf(self, x):
        """Calculate the error function for a given x-value."""
        # Coefficients for the approximation
        a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1 / (1 + 0.3275911 * x)
        poly = (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0])
        return sign * (1 - poly * self.exp(-x * x))

    def cdf(self, x):
        """Calculate the value of the CDF for a given x-value."""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self.erf(z))
