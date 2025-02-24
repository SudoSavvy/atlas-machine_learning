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
            D = np.random.rand(*A.shape) < keep_prob
            A *= D  # Apply dropout
            A /= keep_prob  # Scale activation values
            cache[f"D{l}"] = D  # Store dropout mask
        
        cache[f"A{l}"] = A  # Store activation output
    
    return cache

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using gradient descent.
    
    Parameters:
    Y (np.ndarray): One-hot numpy array of shape (classes, m) with correct labels
    weights (dict): Dictionary containing weights and biases of the network
    cache (dict): Dictionary containing outputs and dropout masks for each layer
    alpha (float): Learning rate
    keep_prob (float): Probability of keeping a node active during dropout
    L (int): Number of layers in the network
    
    Returns:
    None (weights are updated in place)
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # Gradient of softmax loss
    
    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        W = weights[f"W{l}"]
        b = weights[f"b{l}"]
        
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        if l > 1:
            dA_prev = np.dot(W.T, dZ)
            D = cache[f"D{l-1}"]
            dA_prev *= D  # Apply dropout mask
            dA_prev /= keep_prob  # Scale the gradient
            dZ = dA_prev * (A_prev > 0)  # ReLU derivative
        
        # Update weights and biases
        weights[f"W{l}"] -= alpha * dW
        weights[f"b{l}"] -= alpha * db
