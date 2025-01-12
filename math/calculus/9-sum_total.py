#!/usr/bin/env python3

"""
Module: 9-sum_total
Description: This module contains a function to calculate
the sum of squares of the first n integers.
The sum is calculated using the formula:
    sum = n * (n + 1) * (2 * n + 1) / 6
"""


def summation_i_squared(n):
    """
    Function: summation_i_squared
    Description: Calculates the sum of squares of the first n integers.
    Formula used: n * (n + 1) * (2 * n + 1) / 6

    Parameters:
    n (int): The number up to which the sum of squares is calculated.

    Returns:
    int: The sum of squares of the integers from 1 to n.
    None: If the input n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
