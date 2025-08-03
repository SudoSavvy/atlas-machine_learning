#!/usr/bin/env python3
"""
Calculate n-gram BLEU score
"""

from collections import Counter
from math import exp


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Args:
        references (list of list of str): Reference translations
        sentence (list of str): Candidate sentence
        n (int): Size of the n-gram to use

    Returns:
        float: n-gram BLEU score
    """
    def get_ngrams(tokens, n):
        """Generate list of n-grams from a list of tokens"""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # Edge case: empty or too short
    sent_ngrams = get_ngrams(sentence, n)
    if not sent_ngrams:
        return 0.0

    # Count n-grams in candidate
    cand_counter = Counter(sent_ngrams)

    # Build reference max n-gram counts
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_counter = Counter(ref_ngrams)
        for ng in ref_counter:
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0),
                                     ref_counter[ng])

    # Clipped counts
    clipped = 0
    total = 0
    for ng in cand_counter:
        count = cand_counter[ng]
        clipped += min(count, max_ref_counts.get(ng, 0))
        total += count

    precision = clipped / total

    # Brevity penalty
    c = len(sentence)
    r = min((len(ref), abs(len(ref) - c)) for ref in references)[0]

    if c > r:
        bp = 1
    else:
        bp = exp(1 - r / c)

    return bp * precision
