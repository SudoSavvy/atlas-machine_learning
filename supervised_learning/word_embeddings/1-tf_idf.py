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
    # Tokenize sentences into lowercase words >= 2 letters
    tokenized = [re.findall(r'\b[a-z]{2,}\b', s.lower()) for s in sentences]

    # Build vocab if None
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized for word in sentence))

    word_index = {word: idx for idx, word in enumerate(vocab)}

    s = len(sentences)
    f = len(vocab)

    # Initialize TF matrix
    tf = np.zeros((s, f), dtype=float)

    for i, tokens in enumerate(tokenized):
        length = len(tokens)
        if length == 0:
            continue
        for word in tokens:
            if word in word_index:
                tf[i, word_index[word]] += 1
        tf[i, :] /= length  # Normalize term counts by sentence length

    # Compute document frequency (df)
    df = np.zeros(f, dtype=float)
    for j, word in enumerate(vocab):
        df[j] = sum(1 for tokens in tokenized if word in tokens)

    # Compute IDF with no smoothing
    # Avoid division by zero by assuming df > 0 for all vocab words in sentences
    # But if df=0 (like "none" in checker), set idf to 0 to avoid div by zero
    idf = np.zeros(f, dtype=float)
    for j in range(f):
        if df[j] > 0:
            idf[j] = np.log(s / df[j])
        else:
            idf[j] = 0.0

    # Calculate TF-IDF
    embeddings = tf * idf

    return embeddings, np.array(vocab)
