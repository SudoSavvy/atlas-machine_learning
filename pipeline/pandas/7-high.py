#!/usr/bin/env python3
"""
This module defines a function that sorts a pandas DataFrame
by the 'High' column in descending order.
"""

def high(df):
    """
    Sorts the DataFrame by the 'High' column in descending order.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing a 'High' column.

    Returns:
    pd.DataFrame: The sorted DataFrame with rows ordered by 'High' descending.
    """
    return df.sort_values(by='High', ascending=False)
