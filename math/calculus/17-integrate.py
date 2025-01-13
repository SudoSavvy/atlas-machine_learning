#!/usr/bin/env python3

"""
Module: 17-integrate
Description: Computes the indefinite integral of a polynomial.
"""

def poly_integral(poly, C=0):
    """
    Calculate the indefinite integral of a polynomial.

    Parameters:
    poly (list): Coefficients representing the polynomial.
                 Example: [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.
    C (int): The integration constant.

    Returns:
    list: Coefficients of the integral.
    None: If the input is invalid.
    """
    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integral = [C]  # Start with the integration constant
    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        integral.append(int(new_coef) if new_coef.is_integer() else new_coef)

    return integral

# Example usage
if __name__ == "__main__":
    poly = [5, 3, 0, 1]  # Represents f(x) = x^3 + 3x + 5
    print(poly_integral(poly))  # Output: [0, 5, 1.5, 0, 0.25]
