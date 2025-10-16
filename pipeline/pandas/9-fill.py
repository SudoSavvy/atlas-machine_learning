#!/usr/bin/env python3
"""
This module defines a function that cleans a pandas DataFrame by:
- Removing the 'Weighted_Price' column.
- Filling missing values in 'Close' with the previous row's value.
- Filling missing values in 'High', 'Low', and 'Open' with the same row's
 'Close' value.
- Setting missing values in 'Volume_(BTC)' and 'Volume_(Currency)' to 0.
"""


def fill(df):
    """
    Cleans the DataFrame by removing and filling missing values.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing columns:
                       'Weighted_Price', 'Close', 'High', 'Low', 'Open',
                       'Volume_(BTC)', and 'Volume_(Currency)'.

    Returns:
    pd.DataFrame: The modified DataFrame with missing values handled as specified.
    """
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].fillna(method='ffill')

    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    return df
