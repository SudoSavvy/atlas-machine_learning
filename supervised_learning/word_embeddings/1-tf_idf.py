#!/usr/bin/env python3
"""
Creates a TF-IDF embedding matrix from sentences.
"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): Vocabulary words to use. If None,
            all words from sentences are used.

    Returns:
        tuple:
            embeddings (numpy.ndarray): Shape (s, f) TF-IDF matrix where s is
                number of sentences and f number of features.
            features (numpy.ndarray): List of features used for embeddings.
    """
    # Tokenize sentences
    tokenized = [re.findall(r'\b[a-z]{2,}\b', s.lower()) for s in sentences]

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized for word in sentence))

    word_index = {word: idx for idx, word in enumerate(vocab)}
    s = len(sentences)
    f = len(vocab)

    tf = np.zeros((s, f), dtype=float)

    # Raw term frequency
    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_index:
                tf[i, word_index[word]] += 1

    # Document frequency
    df = np.zeros(f, dtype=float)
    for j, word in enumerate(vocab):
        df[j] = sum(1 for tokens in tokenized if word in tokens)

    # Smoothed IDF
    idf = np.log(1 + s / (df + 1e-10))  # epsilon to avoid division by zero

    # TF-IDF
    embeddings = tf * idf

    # L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings /= norms

    return embeddings, np.array(vocab)