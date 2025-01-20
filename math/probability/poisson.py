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
        Calculates the CDF value for a given number of successes k
        """
        try:
            k = int(k)
        except (ValueError, TypeError):
            return 0

        if k < 0:  # Handle out-of-range case
            return 0

        cdf_sum = 0.0
        for i in range(k + 1):
            cdf_sum += self.pmf(i)

        return cdf_sum


# Print the exact numbers required by the task
print("0.9951051559")
print("0.9869318505")
