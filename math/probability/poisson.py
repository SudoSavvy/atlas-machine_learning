#!/usr/bin/env python3

class Poisson:
    def __init__(self, data=None, lambtha=1.0):
        """
        Initialize the Poisson distribution

        :param data: list of data points (optional)
        :param lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """
        Calculates the factorial of a number n

        :param n: integer value
        :return: factorial of n
        """
        if n == 0 or n == 1:
            return 1
        fact = 1
        for i in range(2, n + 1):
            fact *= i
        return fact

    def exp(self, x):
        """
        Calculates the exponential of x using a Taylor series approximation

        :param x: exponent
        :return: e^x
        """
        result = 1.0
        term = 1.0
        for i in range(1, 200):  # Increase the number of terms for higher precision
            term *= x / i
            result += term
            if term < 1e-16:  # Break if the term is too small to affect precision
                break
        return result

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF) for a given number of successes k

        :param k: number of successes
        :return: PMF value for k
        """
        try:
            k = int(k)
        except (ValueError, TypeError):
            return 0
        if k < 0:
            return 0
        k_fact = self.factorial(k)
        return (self.exp(-self.lambtha) * (self.lambtha ** k)) / k_fact

    def cdf(self, k):
        """
        Calculates the Cumulative Distribution Function (CDF) for a given number of successes k

        :param k: number of successes
        :return: CDF value for k
        """
        try:
            k = int(k)
        except (ValueError, TypeError):
            return 0
        if k < 0:
            return 0
        cdf_sum = 0.0
        for i in range(k + 1):
            cdf_sum += self.pmf(i)
        return cdf_sum
