#!/usr/bin/env python3
"""
This module provides a function to concatenate two pandas DataFrames
with a MultiIndex rearranged so that 'Timestamp' is the first level.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Index two DataFrames on their 'Timestamp' columns, filter both to the
      range
    1417411980 to 1417417980 inclusive, concatenate them with keys, and
      rearrange
    the MultiIndex so that 'Timestamp' is the first level.

    Parameters:
        df1 (pd.DataFrame): The coinbase DataFrame.
        df2 (pd.DataFrame): The bitstamp DataFrame.

    Returns:
        pd.DataFrame: A concatenated DataFrame with MultiIndex ('Timestamp',
          'exchange'),
                      sorted in chronological order.
    """
    # Index both DataFrames using the provided index function
    df1 = index(df1)
    df2 = index(df2)

    # Filter both DataFrames to the specified timestamp range
    df1_filtered = df1.loc[1417411980:1417417980]
    df2_filtered = df2.loc[1417411980:1417417980]

    # Concatenate with keys
    combined = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp','coinbase'])

    # Swap MultiIndex levels so Timestamp is first
    combined = combined.swaplevel(0, 1)

    # Sort by Timestamp to ensure chronological order
    combined.sort_index(inplace=True)

    return combined
