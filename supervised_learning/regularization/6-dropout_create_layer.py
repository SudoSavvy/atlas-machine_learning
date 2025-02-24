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
        Z = np.matmul(W, A_prev) + b
        
        if l == L:
            # Softmax activation for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh activation function for hidden layers
            A = np.tanh(Z)
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
    Y (np.ndarray): One-hot numpy.ndarray of shape (classes, m) containing correct labels
    weights (dict): Dictionary containing weights and biases of the network
    cache (dict): Dictionary containing outputs and dropout masks for each layer
    alpha (float): Learning rate
    keep_prob (float): Probability that a node will be kept
    L (int): Number of layers in the network
    
    Updates weights in place.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # Gradient of softmax cross-entropy loss
    
    for l in range(L, 0, -1):
        A_prev = cache[f"A{l - 1}"]
        W = weights[f"W{l}"]
        
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            dA = np.matmul(W.T, dZ)
            dA *= cache[f"D{l - 1}"]  # Apply dropout mask
            dA /= keep_prob  # Scale activation values
            dZ = dA * (1 - np.tanh(cache[f"A{l - 1}"]) ** 2)  # Derivative of tanh
        
        weights[f"W{l}"] -= alpha * dW
        weights[f"b{l}"] -= alpha * db
