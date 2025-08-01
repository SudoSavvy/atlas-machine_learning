#!/usr/bin/env python3
"""Convert a gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a trainable Keras Embedding layer.

    Args:
        model (gensim.models.Word2Vec): A trained Word2Vec model.

    Returns:
        tf.keras.layers.Embedding: A trainable Keras Embedding layer.
    """
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape

    return tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )