#!/usr/bin/env python3
"""Bag of Words Embedding"""


import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences (list): List of sentences to analyze
        vocab (list): Vocabulary words to use for analysis

    Returns:
        embeddings (np.ndarray): (s, f) matrix with embeddings
        features (list): list of features used for embeddings
    """
    # Normalize and tokenize sentences
    all_tokens = []
    for sentence in sentences:
        tokens = re.findall(r'\w+', sentence.lower())
        all_tokens.append(tokens)

    # If vocab not provided, create it from all tokens
    if vocab is None:
        unique_words = sorted(set(word for tokens in all_tokens
                                  for word in tokens))
    else:
        unique_words = sorted(set(vocab))

    # Map words to indices
    word_index = {word: idx for idx, word in enumerate(unique_words)}

    # Initialize embedding matrix
    embeddings = np.zeros((len(sentences), len(unique_words)), dtype=int)

    # Fill embedding matrix
    for i, tokens in enumerate(all_tokens):
        for word in tokens:
            if word in word_index:
                embeddings[i][word_index[word]] += 1

    return embeddings, unique_words
