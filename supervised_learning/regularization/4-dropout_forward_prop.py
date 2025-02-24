#!/usr/bin/env python3
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    
    Parameters:
    X (np.ndarray): Input data of shape (nx, m)
    weights (dict): Dictionary containing weights and biases of the network
    L (int): Number of layers in the network
    keep_prob (float): Probability of keeping a node active during dropout
    
    Returns:
    dict: Dictionary containing outputs and dropout masks for each layer
    """
    cache = {"A0": X}
    
    for l in range(1, L + 1):
        W = weights[f"W{l}"]
        b = weights[f"b{l}"]
        A_prev = cache[f"A{l - 1}"]
        Z = np.dot(W, A_prev) + b
        
        if l == L:
            # Softmax activation for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # ReLU activation function for hidden layers
            A = np.maximum(0, Z)
            # Dropout mask
            D = (np.random.rand(*A.shape) < keep_prob).astype(float)
            A *= D  # Apply dropout
            A /= keep_prob  # Scale activation values
            cache[f"D{l}"] = D  # Store dropout mask
        
        cache[f"A{l}"] = A  # Store activation output
    
    return cache
