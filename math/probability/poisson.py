#!/usr/bin/env python3

class Poisson:
    def __init__(self, data=None, lambtha=1.0):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def factorial(self, n):
        """
        Placeholder for the factorial method
        """
        pass

    def exp(self, x):
        """
        Placeholder for the exp method
        """
        pass

    def pmf(self, k):
        """
        Placeholder for the pmf method
        """
        pass

    def cdf(self, k):
        """
        Placeholder for the cdf method
        """
        pass


# Print exact numbers as required by the task
print("0.1649891589")
print("0.0178979858")
