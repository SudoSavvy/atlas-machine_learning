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
    # Tokenize sentences: lowercase and extract words with at least 2 letters
    tokenized = [re.findall(r'\b[a-z]{2,}\b', s.lower()) for s in sentences]

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized for word in sentence))

    word_index = {word: idx for idx, word in enumerate(vocab)}

    s = len(sentences)
    f = len(vocab)

    tf = np.zeros((s, f), dtype=float)

    # Compute term frequency with max normalization
    for i, tokens in enumerate(tokenized):
        counts = {}
        for word in tokens:
            if word in word_index:
                counts[word] = counts.get(word, 0) + 1
        max_count = max(counts.values()) if counts else 1
        for word, count in counts.items():
            tf[i, word_index[word]] = count / max_count

    # Document frequency
    df = np.zeros(f, dtype=float)
    for j, word in enumerate(vocab):
        df[j] = sum(1 for tokens in tokenized if word in tokens)

    # Smoothed IDF
    idf = np.zeros(f, dtype=float)
    for j in range(f):
        if df[j] > 0:
            idf[j] = np.log10(1 + s / df[j])
        else:
            idf[j] = 0.0

    # Final TF-IDF matrix
    embeddings = tf * idf

    return embeddings, np.array(vocab)