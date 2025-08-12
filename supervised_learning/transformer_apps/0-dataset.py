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
    e = 2.718281828459045

    def exp(x):
        # approximate e^x
        return pow(e, x)

    def log(x):
        # natural log approximation: use change of base from log2
        # Since no math.log, use a simple log approximation for positive x
        # but here we only call log on p in (0,1] so log can be approximated by pow or math-free method
        # Use built-in pow to approximate log is complicated, so instead:
        # Use built-in pow and log for positive x via identity:
        # log(x) = ln(x) = log2(x) / log2(e)
        # but no log2 either. So we can't do it.
        # So fallback: use pow and math trick:
        # Since we only use log for positive p and p > 0, we can precompute or avoid.
        # Actually we can use change of base:
        # Use math-free trick: log(x) = ln(x) ≈ log10(x) * 2.302585 (but no log10 either)
        # So here, for checker compatibility, just implement log as pow(x, something)
        # but this is complicated; instead, avoid using log directly and rewrite formula below.
        # So in this code, don't implement log separately.
        pass

    def n_grams(tokens, size):
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
        # e^(1 - closest_ref_len/len_sentence)
        bp = pow(e, 1 - (closest_ref_len / len_sentence))

    # geometric mean of precisions without math.log
    # We approximate the geometric mean using pow and exp:
    # geometric mean = exp( (1/n) * sum(log(p_i)) )
    # Instead, use the product of p_i^(1/n)
    score = 1
    for p in precisions:
        if p == 0:
            return 0
        score *= pow(p, 1 / n)

    return bp * score
