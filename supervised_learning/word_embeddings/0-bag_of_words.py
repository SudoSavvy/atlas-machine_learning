#!/usr/bin/env python3
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix.

    Args:
        sentences: list of sentences (strings)
        vocab: optional list of vocabulary words. If None, uses all words in sentences.

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: numpy.ndarray of features used (vocabulary words)
    """
    tokenized = [re.findall(r'\b\w+\b', s.lower()) for s in sentences]

    if vocab is None:
        vocab = sorted(set(word for tokens in tokenized for word in tokens))

    word_index = {word: i for i, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, np.array(vocab)
