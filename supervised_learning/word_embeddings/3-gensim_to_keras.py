#!/usr/bin/env python3
"""Yep, still don't know what I'm doing"""

import tensorflow as tf


def gensim_to_keras(model):
    """Here's some documentation"""

    keyed_vectors = model.wv
    weights = keyed_vectors.vectors

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
    )

    return layer