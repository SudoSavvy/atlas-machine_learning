#!/usr/bin/env python3
""" module containing function that creates and trains a FastText model """
import gensim as gm


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """ function that creates and trains a FastText model """
    if cbow:
        sg = 0
    else:
        sg = 1
    fasttext = gm.models.FastText(
        vector_size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=iterations)
    fasttext.build_vocab(sentences)
    fasttext.train(
        sentences,
        total_examples=len(sentences),
        epochs=fasttext.epochs)
    return fasttext