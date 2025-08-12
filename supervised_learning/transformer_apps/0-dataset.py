#!/usr/bin/env python3
"""Dataset Module"""
import tensorflow_datasets as tfds
import transformers


class Dataset():
    """Handles loading and preprocessing of a Portuguese-English translation dataset."""

    def __init__(self):
        """Loads the dataset splits and prepares tokenizers.

        - Downloads and loads the 'ted_hrlr_translate/pt_to_en' dataset.
        - Separates the training and validation sets.
        - Creates tokenizers for both Portuguese and English using the training data.
        - Prints a confirmation message upon successful loading.
        """
        # load dataset
        data, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )
        # separate sets
        self.data_train = data['train']
        self.data_valid = data['validation']

        # build tokenizers from training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        # print if loaded
        print("Datasets correctly loaded")

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers from the given dataset.

        Args:
            data (tf.data.Dataset): Dataset of (Portuguese sentence, English sentence) pairs.

        Returns:
            tokenizer_pt: Tokenizer trained on Portuguese sentences.
            tokenizer_en: Tokenizer trained on English sentences.

        Process:
            - Converts the dataset to iterators of decoded string sentences.
            - Uses pretrained BERT tokenizers as base models.
            - Trains new vocabularies on the dataset with a vocabulary size of 8192.
        """
        # convert datasets to string iterators
        pt_corpus = (pt.decode('utf-8') for pt, _ in data.as_numpy_iterator())
        en_corpus = (en.decode('utf-8') for _, en in data.as_numpy_iterator())

        # initialize and train tokenizers from pretrained models
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        ).train_new_from_iterator(pt_corpus, vocab_size=2**13)

        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        ).train_new_from_iterator(en_corpus, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en
