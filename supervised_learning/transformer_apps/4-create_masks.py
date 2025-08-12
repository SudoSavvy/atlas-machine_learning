#!/usr/bin/env python3
"""Create masks for transformer training and validation"""
import tensorflow as tf

def create_masks(inputs, target):
    """
    Creates encoder padding mask, combined mask for first decoder attention block, 
    and decoder padding mask for second decoder attention block.

    Args:
        inputs (tf.Tensor): shape (batch_size, seq_len_in), input sentence tokens
        target (tf.Tensor): shape (batch_size, seq_len_out), target sentence tokens

    Returns:
        encoder_mask (tf.Tensor): shape (batch_size, 1, 1, seq_len_in), padding mask for encoder
        combined_mask (tf.Tensor): shape (batch_size, 1, seq_len_out, seq_len_out), combined padding + lookahead mask for decoder first attention block
        decoder_mask (tf.Tensor): shape (batch_size, 1, 1, seq_len_in), padding mask for decoder second attention block
    """

    # Encoder padding mask (mask all padding tokens in inputs)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len_in)

    # Decoder padding mask (mask all padding tokens in inputs for 2nd attention block)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len_in)

    # Decoder target padding mask
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(target)[1], tf.shape(target)[1])), -1, 0)
    # Shape (seq_len_out, seq_len_out)
    # look_ahead_mask has 1s in upper triangle, masking future tokens

    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)  # (batch_size, seq_len_out)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len_out)

    # Broadcast look_ahead_mask to (1, 1, seq_len_out, seq_len_out)
    look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]

    # combined mask: maximum of lookahead mask and target padding mask (broadcast last dim)
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
