#!/usr/bin/env python3

class Exponential:
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.

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
            # Estimate lambtha as the inverse of the mean of the data
            self.lambtha = 1 / (sum(data) / len(data))


    def pdf(self, x):
        """
        Calculates the Probability Density Function (PDF) for a given time period x.

        :param x: time period
        :return: PDF value for x
        """
        if x < 0:
            return 0
        # Calculate e^(-lambda * x) manually
        exp_value = 1
        term = 1
        n = 1
        if self.lambtha * x > 700:  # Prevent overflow in large exponentials
            return 0
        while True:
            term *= (-self.lambtha * x) / n
            exp_value += term
            n += 1
            if abs(term) < 1e-15:  # Stop when the term becomes small enough
                break
        return self.lambtha * exp_value


    def cdf(self, x):
        """
        Calculates the Cumulative Distribution Function (CDF) for a given time period x.

        :param x: time period
        :return: CDF value for x
        """
        if x < 0:
            return 0
        # Calculate CDF: 1 - e^(-lambtha * x)
        exp_value = 1
        term = 1
        n = 1
        if self.lambtha * x > 700:  # Prevent overflow for large exponentials
            return 1
        while True:
            term *= (-self.lambtha * x) / n
            exp_value += term
            n += 1
            if abs(term) < 1e-15:  # Stop when the term becomes small enough
                break
        return 1 - exp_value
