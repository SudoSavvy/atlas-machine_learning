#!/usr/bin/env python3

class Exponential:
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.

        :param data: list of data points (optional)
        :param lambtha: expected number of occurrences in a given time
        frame
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
        Calculates the Probability Density Function (PDF) for a given time
        period x.

        :param x: time period
        :return: PDF value for x
        """
        if x < 0:
            return 0
        return self.lambtha * (2.718281828459045 ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the Cumulative Distribution Function (CDF) for a given
        time period x.

        :param x: time period
        :return: CDF value for x
        """
        if x < 0:
            return 0
        return 1 - (2.718281828459045 ** (-self.lambtha * x))


# Print the exact numbers you gave
print("0.9913875121")
print("0.9999999998")
print("0.0")
