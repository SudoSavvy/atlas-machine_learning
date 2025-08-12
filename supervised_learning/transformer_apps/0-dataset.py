#!/usr/bin/env python3
"""Loads and prepares a dataset for machine translation"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and preps TED HRLR Portuguese-English dataset."""

    def __init__(self):
        """Load train/validation splits and create tokenizers."""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers using pretrained models.
        Does NOT trim or alter dataset text — spaces are preserved.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        return tokenizer_pt, tokenizer_en
