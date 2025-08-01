#!/usr/bin/env python3
"""
Module to create TF-IDF embeddings from sentences.
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
                the number of sentences and f is the number of features.
            features (numpy.ndarray): List of features (vocabulary words) used
                for embeddings.
    """
    # Tokenize sentences (lowercase, words of length >=2)
    tokenized = [re.findall(r'\b[a-z]{2,}\b', s.lower()) for s in sentences]

    # Build vocab if None
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized for word in sentence))

    word_index = {word: idx for idx, word in enumerate(vocab)}
    s = len(sentences)
    f = len(vocab)

    # Compute term frequency (TF) matrix: counts per sentence
    tf = np.zeros((s, f), dtype=float)
    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_index:
                tf[i, word_index[word]] += 1
        # Normalize TF by number of tokens in sentence to get frequency
        if len(tokens) > 0:
            tf[i, :] /= len(tokens)

    # Compute document frequency (DF) per term
    df = np.zeros(f, dtype=float)
    for j in range(f):
        # Count how many sentences contain vocab[j]
        df[j] = sum(1 for tokens in tokenized if vocab[j] in tokens)

    # Compute inverse document frequency (IDF)
    # Adding 1 to denominator for smoothing to avoid division by zero
    idf = np.log((s + 1) / (df + 1)) + 1

    # Compute TF-IDF
    embeddings = tf * idf

    return embeddings, np.array(vocab)
