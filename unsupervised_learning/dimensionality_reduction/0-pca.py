#!/usr/bin/env python3
import numpy as np
pca = __import__('1-pca').pca

# Load dataset
data = np.load('data.npz')
X = data['X']

# Perform PCA to keep 99% variance
W = pca(X, var=0.99)

# Project the data
X_transformed = np.matmul(X - np.mean(X, axis=0), W)

# Print transformed data and shape
print(X_transformed)
print(X_transformed.shape)

# For comparison: project with only first component
W1 = pca(X, var=0.5)
X1 = np.matmul(X - np.mean(X, axis=0), W1)
print(X1)
print(X1.shape)
