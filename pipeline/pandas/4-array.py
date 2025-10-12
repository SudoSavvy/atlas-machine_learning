#!/usr/bin/env python3
"""
This module defines a function that extracts the last 10 rows
from the 'High' and 'Close' columns of a pandas DataFrame
and converts them into a NumPy ndarray.
"""


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    from the given DataFrame and converts them into a NumPy ndarray.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing 'High' and 'Close' columns.

    Returns:
    numpy.ndarray: A NumPy array containing the last 10 rows of the selected columns.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
