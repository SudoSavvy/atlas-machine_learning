#!/usr/bin/env python3
"""
Preprocess BTC time series data from Coinbase and Bitstamp
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.sort_values('Timestamp')
    return df

def merge_sources(coinbase_path, bitstamp_path):
    df_cb = load_and_clean(coinbase_path)
    df_bs = load_and_clean(bitstamp_path)
    
    df = pd.merge(df_cb, df_bs, on="Timestamp", suffixes=('_cb', '_bs'))
    df['close_avg'] = (df['Close_cb'] + df['Close_bs']) / 2
    df = df[['Timestamp', 'close_avg']]
    return df

def create_sequences(data, window=24*60, forecast=60):
    X, y = [], []
    for i in range(len(data) - window - forecast):
        X.append(data[i:i+window])
        y.append(data[i+window+forecast-1])  # target: 1 hour after window
    return np.array(X), np.array(y)

def preprocess():
    df = merge_sources('coinbase.csv', 'bitstamp.csv')
    
    scaler = MinMaxScaler()
    df['close_scaled'] = scaler.fit_transform(df[['close_avg']])
    
    X, y = create_sequences(df['close_scaled'].values)

    np.savez_compressed('btc_data.npz', X=X, y=y)
    print("Preprocessing complete. Saved to btc_data.npz.")

if __name__ == '__main__':
    preprocess()
