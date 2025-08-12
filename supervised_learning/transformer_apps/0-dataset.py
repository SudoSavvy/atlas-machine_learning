#!/usr/bin/env python3
"""
Dataset class for loading and preparing TED Talks
Portuguese to English translation dataset
"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Loads and prepares TED Talks pt→en dataset for translation
    """

    def __init__(self):
        """
        Class constructor: loads training and validation datasets
        and creates tokenizers
        """
        # Load TED Talks translation dataset as supervised tuples (pt, en)
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

        # Create tokenizers from training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained models

        Args:
            data (tf.data.Dataset): dataset containing (pt, en) sentence pairs

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Pre-trained tokenizers
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        # The vocab size is already fixed in the pretrained models
        # so we don't need to train them from scratch here.
        # However, if needed, training could be added with max vocab size 2**13

        return tokenizer_pt, tokenizer_en
