#!/usr/bin/env python3
"""
Bag of Words Embedding Function
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences (list): List of sentences to analyze
        vocab (list): List of vocabulary words to use for analysis.
                      If None, uses all unique words in the sentences.

    Returns:
        embeddings (np.ndarray): Shape (s, f) containing the embeddings
            - s is the number of sentences
            - f is the number of features (words)
        features (list): List of features (words) used for the embeddings
    """
    # Function to tokenize and normalize words
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    # Tokenize all sentences
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    # Build vocabulary if not provided
    if vocab is None:
        unique_words = set()
        for tokens in tokenized_sentences:
            unique_words.update(tokens)
        features = sorted(unique_words)
    else:
        features = sorted(set(vocab))

    # Initialize the embedding matrix
    embeddings = np.zeros((len(sentences), len(features)))

    # Create a word index mapping
    word_idx = {word: i for i, word in enumerate(features)}

    # Fill in the embedding matrix
    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_idx:
                embeddings[i, word_idx[word]] += 1

    return embeddings, features
