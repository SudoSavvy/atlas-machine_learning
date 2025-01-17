import math

class Poisson:
    def __init__(self, data=None, lambtha=1.):
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

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF) for a given number of successes k

        :param k: number of successes
        :return: PMF value for k
        """
        if k < 0 or not isinstance(k, int):
            return 0
        k_fact = math.factorial(k)
        return (math.exp(-self.lambtha) * (self.lambtha ** k)) / k_fact

    def cdf(self, k):
        """
        Calculates the Cumulative Distribution Function (CDF) for a given number of successes k

        :param k: number of successes
        :return: CDF value for k
        """
        if k < 0 or not isinstance(k, int):
            return 0
        cdf_sum = 0
        for i in range(k + 1):
            cdf_sum += self.pmf(i)
        return cdf_sum
