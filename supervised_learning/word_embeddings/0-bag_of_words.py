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
        features (np.ndarray): Array of features (words) used for the embeddings
    """
    def tokenize(text):
        """Tokenizes a sentence into lowercase words"""
        return re.findall(r'\b\w+\b', text.lower())

    # Tokenize all sentences
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    # Determine features
    if vocab is None:
        unique_words = set()
        for tokens in tokenized_sentences:
            unique_words.update(tokens)
        features = sorted(unique_words)
    else:
        features = sorted(set(vocab))

    # Convert features list to numpy array to match expected output
    features = np.array(features)

    # Initialize embedding matrix with integer type
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    # Create word index map
    word_to_index = {word: i for i, word in enumerate(features)}

    # Fill in embeddings
    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, features
