#!/usr/bin/env python3
"""
Calculate unigram BLEU score
"""

from math import exp, log
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list of list of str): Reference translations
        sentence (list of str): Candidate sentence

    Returns:
        float: Unigram BLEU score
    """
    sentence_len = len(sentence)
    if sentence_len == 0:
        return 0.0

    # Count unigrams in candidate
    cand_counter = Counter(sentence)

    # Count max reference unigram frequencies
    max_ref_counts = {}
    for ref in references:
        ref_counter = Counter(ref)
        for word in ref_counter:
            max_ref_counts[word] = max(max_ref_counts.get(word, 0),
                                       ref_counter[word])

    # Clipped unigram counts
    clipped_count = 0
    for word in cand_counter:
        count = min(cand_counter[word], max_ref_counts.get(word, 0))
        clipped_count += count

    precision = clipped_count / sentence_len

    # Find reference length closest to candidate length
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens,
                          key=lambda r: (abs(r - sentence_len), r))

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = exp(1 - closest_ref_len / sentence_len)

    bleu_score = bp * precision
    return bleu_score
