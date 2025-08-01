#!/usr/bin/env python3
"""Module for creating bag of words embeddings."""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    # Tokenize: convert to lowercase and split on word boundaries,
    # excluding 's' and similar
    tokenized = [re.findall(r'\b[a-z]{2,}\b', s.lower()) for s in sentences]

    if vocab is None:
        # Flatten and sort unique words
        vocab = sorted(
            set(word for sentence in tokenized for word in sentence)
        )

    word_index = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_index:
                embeddings[i][word_index[word]] += 1

    return embeddings, np.array(vocab)
