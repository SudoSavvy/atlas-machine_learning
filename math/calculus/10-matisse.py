#!/usr/bin/env python3

"""
Module: 10-matisse
Description: This module contains a function to calculate the
derivative of a polynomial represented as a list of
coefficients.
The index of the list represents the power of x for the
corresponding coefficient.
"""


def poly_derivative(poly):
    """
    Function: poly_derivative
    Description: Calculates the derivative of a polynomial represented
    as a list of coefficients.

    Parameters:
    poly (list): A list of coefficients where the index represents the
    power of x.

    Returns:
    list: A new list of coefficients representing the derivative of
    the polynomial.
    None: If the input is not a valid list of coefficients.

    Notes:
    - If the derivative is 0, the function returns [0].
    - Invalid inputs include non-lists, empty lists, or lists containing
    non-numeric elements.
    """
    if not isinstance(poly, list) or not poly or not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Derive the polynomial
    derivative = [i * poly[i] for i in range(1, len(poly))]

    # Return [0] if the derivative is zero (i.e., original polynomial is a constant)
    return derivative if derivative else [0]
