#!/usr/bin/env python3


def index(df):
    """
    Set the 'Timestamp' column as the index of the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Timestamp' as its index.
    """
    df.set_index('Timestamp', inplace=True)
    return df
