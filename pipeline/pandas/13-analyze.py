#!/usr/bin/env python3
"""
This module provides a function to compute descriptive statistics
for all columns in a DataFrame except the 'Timestamp' column.
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except 'Timestamp'.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'Timestamp
          column.

    Returns:
        pd.DataFrame: A new DataFrame with descriptive statistics for
                      all columns except 'Timestamp'.
    """
    # Drop the 'Timestamp' column if it exists
    df_no_timestamp = df.drop(columns=['Timestamp'], errors='ignore')

    # Compute and return descriptive statistics
    return df_no_timestamp.describe()
