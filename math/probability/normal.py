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
        pi = 3.141592653589793
        sqrt_2pi = (2 * pi) ** 0.5

        coeff = 1 / (self.stddev * sqrt_2pi)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)

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

    def exp(self, x):
        """
        Calculate the exponential function for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The exponential function value for x.
        """
        exp_value = 1
        term = 1
        n = 1
        while True:
            term *= x / n
            exp_value += term
            n += 1
            if abs(term) < 1e-15:
                break

        return exp_value

    def erf(self, x):
        """
        Calculate the error function for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The error function value for x.
        """
        a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
        t = 1 / (1 + 0.3275911 * abs(x))
        y = 1 - (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0]) * t * self.exp(-x * x)
        return y if x >= 0 else -y

    def cdf(self, x):
        """
        Calculate the value of the CDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self.erf(z))
