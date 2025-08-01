#!/usr/bin/env python3
"""Train a FastText model using gensim."""

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
    workers=1,
):
    """
    Builds and trains a FastText model using gensim.

    Args:
        sentences (list): A list of tokenized sentences for training.
        vector_size (int): Dimensionality of the word vectors.
        min_count (int): Minimum frequency threshold for words to be included.
        negative (int): Number of negative samples used in training.
        window (int): Maximum distance between the current and predicted word.
        cbow (bool): If True, use CBOW; if False, use Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Random seed for reproducibility.
        workers (int): Number of worker threads to use during training.

    Returns:
        gensim.models.FastText: The trained FastText model.
    """
    model = gensim.models.FastText(
        sentences=sentences,
        min_count=min_count,
        window=window,
        negative=negative,
        cbow_mean=cbow,
        hs=not cbow,
        alpha=0.025,
        min_alpha=0.001,
        seed=seed,
        workers=workers,
        epochs=epochs,
        vector_size=vector_size,
    )

    return model