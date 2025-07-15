#!/usr/bin/env python3
"""
Train and evaluate a model to forecast BTC price using RNNs
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data(path='btc_data.npz'):
    data = np.load(path)
    return data['X'], data['y']

def prepare_dataset(X, y, batch_size=64, split=0.8):
    split_idx = int(len(X) * split)
    train_ds = tf.data.Dataset.from_tensor_slices((X[:split_idx], y[:split_idx]))
    val_ds = tf.data.Dataset.from_tensor_slices((X[split_idx:], y[split_idx:]))
    
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    X, y = load_data()
    train_ds, val_ds = prepare_dataset(X, y)
    model = build_model(X.shape[1:])

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save('btc_forecast_model.h5')
    print("Model saved as btc_forecast_model.h5")

if __name__ == '__main__':
    main()
