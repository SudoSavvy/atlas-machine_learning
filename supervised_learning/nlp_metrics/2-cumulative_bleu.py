#!/usr/bin/env python3
"""
Calculate cumulative n-gram BLEU score
"""

from collections import Counter
from math import log, exp


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references (list of list of str): Reference translations
        sentence (list of str): Candidate sentence
        n (int): Size of the largest n-gram to use

    Returns:
        float: cumulative n-gram BLEU score
    """
    def get_ngrams(tokens, size):
        """Generate list of n-grams from a list of tokens"""
        return [tuple(tokens[i:i + size])
                for i in range(len(tokens) - size + 1)]

    precisions = []
    for i in range(1, n + 1):
        cand_ngrams = get_ngrams(sentence, i)
        if not cand_ngrams:
            precisions.append(0)
            continue

        cand_counter = Counter(cand_ngrams)
        max_ref_counts = {}

        for ref in references:
            ref_ngrams = get_ngrams(ref, i)
            ref_counter = Counter(ref_ngrams)
            for ng in ref_counter:
                max_ref_counts[ng] = max(max_ref_counts.get(ng, 0),
                                         ref_counter[ng])

        clipped = sum(min(cand_counter[ng], max_ref_counts.get(ng, 0))
                      for ng in cand_counter)
        total = sum(cand_counter.values())
        precisions.append(clipped / total if total > 0 else 0)

    # If any precision is 0, BLEU score is 0
    if min(precisions) == 0:
        return 0.0

    # Log average of precisions
    log_precisions = sum(log(p) for p in precisions) / n
    geo_mean = exp(log_precisions)

    # Brevity penalty
    c = len(sentence)
    r = min((len(ref), abs(len(ref) - c)) for ref in references)[0]

    bp = 1 if c > r else exp(1 - r / c)

    return bp * geo_mean
