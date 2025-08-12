#!/usr/bin/env python3
"""Dataset Module"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset():
    """Loads and prepares the Portuguese-English translation dataset."""

    def __init__(self, batch_size, max_len):
        """Load dataset, build tokenizers, tokenize, and set up data pipeline.

        Args:
            batch_size (int): Batch size for training and validation.
            max_len (int): Maximum token length allowed per sentence.
        """
        data, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )
        self.data_train = data['train']
        self.data_valid = data['validation']

        # Build tokenizers from training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Tokenize datasets
        self.data_train = self.data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.data_valid = self.data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Filter function: only keep examples with both sequences <= max_len
        def filter_max_len(pt, en):
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        # Setup training dataset pipeline
        self.data_train = (
            self.data_train
            .filter(filter_max_len)
            .cache()
            .shuffle(buffer_size=20000)
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Setup validation dataset pipeline
        self.data_valid = (
            self.data_valid
            .filter(filter_max_len)
            .padded_batch(batch_size, padded_shapes=([None], [None]))
        )

        print("Datasets correctly loaded")

    def tokenize_dataset(self, data):
        """Create subword tokenizers from the dataset."""
        pt_corpus = (pt.decode('utf-8') for pt, _ in data.as_numpy_iterator())
        en_corpus = (en.decode('utf-8') for _, en in data.as_numpy_iterator())

        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        ).train_new_from_iterator(pt_corpus, vocab_size=2**13)

        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        ).train_new_from_iterator(en_corpus, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode Portuguese and English sentences with start/end tokens.

        Args:
            pt (tf.Tensor): Portuguese sentence tensor
            en (tf.Tensor): English sentence tensor

        Returns:
            pt_tokens (tf.Tensor): Tokenized Portuguese sentence
            en_tokens (tf.Tensor): Tokenized English sentence
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_ids = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        pt_tokens = tf.constant([pt_vocab_size] + pt_ids + [pt_vocab_size + 1], dtype=tf.int64)
        en_tokens = tf.constant([en_vocab_size] + en_ids + [en_vocab_size + 1], dtype=tf.int64)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper around encode method."""
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
