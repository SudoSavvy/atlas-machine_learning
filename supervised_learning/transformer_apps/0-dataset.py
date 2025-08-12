#!/usr/bin/env python3
"""Cumulative n-gram BLEU score calculation without nltk"""

import tensorflow_datasets as tfds
import transformers


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list of list of str): Reference translations
        sentence (list of str): Model-generated sentence
        n (int): Largest n-gram size to use

    Returns:
        float: Cumulative n-gram BLEU score
    """
    # Helper: extract n-grams from a sentence
    def ngrams(tokens, size):
        return [tuple(tokens[i:i+size]) for i in range(len(tokens) - size + 1)]

    # Helper: modified precision
    def modified_precision(references, sentence, size):
        sen_ngrams = ngrams(sentence, size)
        sen_counts = {}
        for ng in sen_ngrams:
            sen_counts[ng] = sen_counts.get(ng, 0) + 1

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = ngrams(ref, size)
            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1
            for ng in sen_counts:
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0),
                                         ref_counts.get(ng, 0))

        clipped_counts = 0
        total_counts = 0
        for ng in sen_counts:
            clipped_counts += min(sen_counts[ng], max_ref_counts.get(ng, 0))
            total_counts += sen_counts[ng]

        if total_counts == 0:
            return 0
        return clipped_counts / total_counts

    # Brevity penalty
    ref_lens = [len(ref) for ref in references]
    sen_len = len(sentence)
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - sen_len), rl))

    if sen_len == 0:
        return 0
    if sen_len > closest_ref_len:
        bp = 1
    else:
        # Implement exp without math.exp
        bp = pow(2.718281828459045, 1 - (closest_ref_len / sen_len))

    # Calculate average log precisions
    precisions = []
    for i in range(1, n + 1):
        p_i = modified_precision(references, sentence, i)
        if p_i == 0:
            precisions.append(-9999999999)  # log(0) → large negative
        else:
            precisions.append(pow(p_i, 1 / n))

    # Multiply instead of log-sum-exp to avoid math.log
    score = bp
    for p in precisions:
        if p <= 0:
            return 0
        score *= p

    return score
