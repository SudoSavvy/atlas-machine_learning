#!/usr/bin/env python3
"""
This module defines a function that removes rows from a pandas DataFrame
where the 'Close' column contains NaN values.
"""

def prune(df):
    """
    Removes rows from the DataFrame where the 'Close' column has NaN values.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing a 'Close' column.

    Returns:
    pd.DataFrame: The modified DataFrame with NaN entries in 'Close' removed.
    """
    return df[df['Close'].notna()]
