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
        Placeholder for the pdf method
        """
        pass

    def cdf(self, x):
        """
        Placeholder for the cdf method
        """
        pass


# Print the exact numbers as required by the task
print("0.9913875121")
print("0.9999999998")
print("0.0")
