#!/usr/bin/env python3
"""Ok this is where the difficulty spikes
I'm gonna slam through this the same way I do everything
Panicking"""

import gensim


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1,
):
    """I officially have no idea what I'm doing"""
    model = gensim.models.Word2Vec(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        cbow_mean=cbow,
        # hs=not cbow,
        # alpha=0.025,
        # min_alpha=0.001,
        seed=seed,
        workers=workers,
        epochs=epochs,
    )

    return model