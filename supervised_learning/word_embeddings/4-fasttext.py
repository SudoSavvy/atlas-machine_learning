#!/usr/bin/env python3
"""Build and train a FastText model using gensim."""

import gensim


def fasttext_model(
    sentences,
    vector_size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a gensim FastText model.

    Args:
        sentences (list): List of tokenized sentences.
        vector_size (int): Dimensionality of the embedding vectors.
        min_count (int): Minimum word frequency threshold.
        negative (int): Number of negative samples.
        window (int): Context window size.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Random seed for reproducibility.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.FastText: The trained FastText model.
    """
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed,
        workers=workers
    )

    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=epochs
    )

    return model