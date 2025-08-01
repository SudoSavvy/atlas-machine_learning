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
    # Sort vocabulary keys to ensure consistent ordering
    sorted_keys = sorted(model.wv.key_to_index, key=model.wv.key_to_index.get)
    weights = [model.wv[word] for word in sorted_keys]

    # Convert to tensor
    embedding_matrix = tf.constant(weights, dtype=tf.float32)
    vocab_size, embedding_dim = embedding_matrix.shape

    return tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )