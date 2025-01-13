#!/usr/bin/env python3

def poly_derivative(poly):
    """
    Function: poly_derivative
    Description: Calculates the derivative of a polynomial represented by a list of coefficients.
    
    Parameters:
    poly (list): A list of coefficients where the index represents the power of x.
                 Example: [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.
    
    Returns:
    list: A list of coefficients representing the derivative of the polynomial.
          Returns [0] if the derivative is 0.
    None: If the input is not a valid list of coefficients.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 0 or len(poly) == 1:  # Constant polynomial or empty list
        return [0]
    
    # Compute the derivative
    derivative = [i * poly[i] for i in range(1, len(poly))]
    
    return derivative if derivative else [0]
