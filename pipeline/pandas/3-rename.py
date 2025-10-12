#!/usr/bin/env python3
"""
This module defines a function that renames the 'Timestamp' column
in a pandas DataFrame to 'Datetime', converts its values to datetime format,
and returns a DataFrame containing only the 'Datetime' and 'Close' columns.
"""

import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime', converts its values to datetime,
    and returns a DataFrame with only the 'Datetime' and 'Close' columns.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing a column named 'Timestamp'
                       and another column named 'Close'.

    Returns:
    pd.DataFrame: A modified DataFrame with 'Datetime' and 'Close' columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ns')
    return df[['Datetime', 'Close']]
