#!/usr/bin/env python3

class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the Binomial distribution."""
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

    def factorial(self, n):
        """Calculates the factorial of n."""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def comb(self, n, k):
        """Calculates the number of combinations C(n, k)."""
        if k > n or k < 0:
            return 0
        return self.factorial(n) // (self.factorial(k) * self.factorial(n - k))

    def pow(self, base, exp):
        """Calculates base raised to the power exp."""
        result = 1
        for _ in range(exp):
            result *= base
        return result

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return self.comb(self.n, k) * self.pow(self.p, k) * self.pow(1 - self.p, self.n - k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: CDF value for k.
        """
        # Ensure k is an integer
        k = int(k)

        # If k is out of range, return 0
        if k < 0:
            return 0
        if k > self.n:
            return 1

        # Sum the PMF values for all values up to and including k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
