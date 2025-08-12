#!/usr/bin/env python3
"""Calculate cumulative n-gram BLEU score for a sentence"""
import tensorflow_datasets as tfds
import transformers


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    Args:
        references: list of reference translations
                    each a list of the words in the translation
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation
    Returns:
        cumulative n-gram BLEU score
    """
    import math

    def n_grams(tokens, size):
        """Return list of n-grams of given size from tokens."""
        return [tuple(tokens[i:i+size]) for i in range(len(tokens) - size + 1)]

    precisions = []
    for i in range(1, n + 1):
        sentence_ngrams = n_grams(sentence, i)
        ref_counts = {}
        for ref in references:
            ref_ng = n_grams(ref, i)
            counts = {}
            for ng in ref_ng:
                counts[ng] = counts.get(ng, 0) + 1
            for ng, cnt in counts.items():
                ref_counts[ng] = max(ref_counts.get(ng, 0), cnt)

        matches = 0
        counts = {}
        for ng in sentence_ngrams:
            counts[ng] = counts.get(ng, 0) + 1
        for ng, cnt in counts.items():
            matches += min(cnt, ref_counts.get(ng, 0))

        precisions.append(matches / len(sentence_ngrams)
                          if sentence_ngrams else 0)

    # brevity penalty
    len_sentence = len(sentence)
    ref_lens = [len(r) for r in references]
    closest_ref_len = min(ref_lens,
                          key=lambda ref_len: (abs(ref_len - len_sentence),
                                               ref_len))
    if len_sentence == 0:
        bp = 0
    elif len_sentence > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len_sentence)

    # geometric mean of precisions
    if min(precisions) > 0:
        score = math.exp(sum((1 / n) * math.log(p) for p in precisions))
    else:
        score = 0

    return bp * score
