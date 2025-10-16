#!/usr/bin/env python3
"""
This script transforms and visualizes cryptocurrency trading data from a CSV file.
It cleans missing values, converts timestamps to dates, aggregates daily statistics,
and plots the results.
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the 'Weighted_Price' column
df.drop(columns=['Weighted_Price'], inplace=True)

# Rename 'Timestamp' to 'Date' and convert to datetime
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Fill missing values
df['Close'].fillna(method='ffill', inplace=True)
for col in ['High', 'Low', 'Open']:
    df[col].fillna(df['Close'], inplace=True)
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col].fillna(0, inplace=True)

# Filter data from 2017 onwards
df = df.loc[df.index >= '2017-01-01']

# Resample to daily intervals and aggregate
daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the daily closing price
daily_df['Close'].plot(title='Daily Closing Price (2017–2019)', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.tight_layout()
plt.show()

# Return the transformed DataFrame
print(daily_df)
