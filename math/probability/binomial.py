#!/usr/bin/env python3

class Binomial:
    """Represents a binomial distribution."""

    # Existing methods (e.g., __init__) go here...

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF) for a given number of successes k.

        Args:
            k (int): Number of successes.

        Returns:
            float: PMF value for k.
        """
        # Ensure k is an integer
        k = int(k)

        # Check if k is within range
        if k < 0 or k > self.n:
            return 0

        # Calculate factorial manually
        def factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

        # Calculate binomial coefficient C(n, k) = n! / (k! * (n - k)!)
        n_factorial = factorial(self.n)
        k_factorial = factorial(k)
        n_k_factorial = factorial(self.n - k)
        binomial_coeff = n_factorial / (k_factorial * n_k_factorial)

        # Calculate PMF
        p_k = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        pmf_value = binomial_coeff * p_k

        return pmf_value
