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
        features: list of features used (vocabulary words)
    """
    tokenized_sentences = [
        re.findall(r'\b\w+\b', sentence.lower())
        for sentence in sentences
    ]

    if vocab is None:
        unique_words = set()
        for tokens in tokenized_sentences:
            unique_words.update(tokens)
        vocab = sorted(unique_words)

    word_to_index = {word: i for i, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_to_index:
                embeddings[i, word_to_index[token]] += 1

    return embeddings, vocab
