#!/usr/bin/env python3

class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        Args:
            data (list): List of data points (optional).
            n (int): Number of trials (must be positive).
            p (float): Probability of success (0 < p < 1).

        Raises:
            ValueError: If n is not a positive value.
            ValueError: If p is not between 0 and 1.
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from the data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Calculate p and n
            p = 1 - (variance / mean)
            n = round(mean / p)

            # Recalculate p using the rounded n
            p = mean / n

            self.n = int(n)
            self.p = float(p)
