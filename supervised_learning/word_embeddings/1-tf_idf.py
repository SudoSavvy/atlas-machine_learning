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
    # Tokenize sentences to lowercase words length >= 2
    tokenized = [re.findall(r'\b[a-z]{2,}\b', s.lower()) for s in sentences]

    # Build vocab if None
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized for word in sentence))

    word_index = {word: idx for idx, word in enumerate(vocab)}
    s = len(sentences)
    f = len(vocab)

    # Compute TF matrix: term frequency = count / number of tokens in sentence
    tf = np.zeros((s, f), dtype=float)
    for i, tokens in enumerate(tokenized):
        length = len(tokens)
        for word in tokens:
            if word in word_index:
                tf[i, word_index[word]] += 1
        if length > 0:
            tf[i, :] /= length

    # Compute document frequency: number of sentences containing the term
    df = np.zeros(f, dtype=float)
    for j, word in enumerate(vocab):
        df[j] = sum(1 for tokens in tokenized if word in tokens)

    # Compute IDF without smoothing
    # Avoid division by zero by assuming df[j] > 0 always (words appear at least once)
    idf = np.log(s / df)

    # Compute TF-IDF
    embeddings = tf * idf

    return embeddings, np.array(vocab)
