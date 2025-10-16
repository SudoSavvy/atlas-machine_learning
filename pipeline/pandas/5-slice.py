#!/usr/bin/env python3
"""
This module defines a function that extracts specific columns from a pandas
DataFrame and selects every 60th row from those columns.
"""


def slice(df):
    """
    Extracts the 'High', 'Low', 'Close', and 'Volume_(BTC)' columns from the
    given DataFrame and returns every 60th row from those columns.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the required columns.

    Returns:
    pd.DataFrame: A sliced DataFrame with every 60th row of the selected
      columns.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
