#!/usr/bin/env python3
"""
This module provides a function to concatenate two pandas DataFrames
after indexing them on their 'Timestamp' columns and filtering one
based on a specific timestamp.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Index two DataFrames on their 'Timestamp' columns, filter df2 up to
    and including timestamp 1417411920, and concatenate the filtered df2
    above df1 with appropriate keys.

    Parameters:
        df1 (pd.DataFrame): The coinbase DataFrame.
        df2 (pd.DataFrame): The bitstamp DataFrame.

    Returns:
        pd.DataFrame: A concatenated DataFrame with keys 'bitstamp' and
          'coinbase'.
    """
    # Index both DataFrames using the provided index function
    df1 = index(df1)
    df2 = index(df2)

    # Filter df2 to include timestamps up to and including 1417411920
    df2_filtered = df2.loc[:1417411920]

    # Concatenate with keys
    result = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

    return result
