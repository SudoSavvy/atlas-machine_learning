#!/usr/bin/env python3
"""Train Word2Vec Module"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """Function that creates , builds and trains a gensim word2vec model:

    sentences is a list of sentences to be trained on
    vector_size is the dimensionality of the embedding layer
    min_count is the minimum number of occurrences of a word for use in
    training
    window is the maximum distance between the current and predicted word
    within a sentence
    negative is the size of negative sampling
    cbow is a boolean to determine the training type; True is for CBOW;
    False is for Skip-gram
    epochs is the number of iterations to train over
    seed is the seed for the random number generator
    workers is the number of worker threads to train the model
    Returns: the trained model

    """
    # init the Word2Vec model with the given parameters
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers
    )

    # call and build
    model.build_vocab(sentences)
    # train the model on the sentences for the given number of epochs
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    # return the trained Word2Vec model
    return model
