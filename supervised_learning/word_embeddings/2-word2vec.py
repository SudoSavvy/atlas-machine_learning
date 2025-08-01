#!/usr/bin/env python3
"""
Creates, builds, and trains a Word2Vec model using gensim.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Trains a Word2Vec model using gensim.

    Args:
        sentences (list of list of str): Tokenized sentences to train on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word to be included.
        window (int): Maximum distance between current and predicted word.
        negative (int): Number of negative samples.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Random seed for reproducibility.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    model.train(sentences, total_examples=len(sentences), epochs=epochs)

    return model