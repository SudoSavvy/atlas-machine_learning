#!/usr/bin/env python3

class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.

        Args:
            data (list, optional): Data points to estimate the distribution.
            n (int): Number of trials (default: 1).
            p (float): Probability of success (default: 0.5).

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has less than two points.
            ValueError: If n is not a positive integer or p is invalid.
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

            # Calculate mean and variance from data
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)

            # Estimate p and n
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)

            # Recalculate p based on the rounded n value
            self.p = mean / self.n
