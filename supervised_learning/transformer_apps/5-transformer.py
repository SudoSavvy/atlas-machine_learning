#!/usr/bin/env python3
"""Transformer Model"""

import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask):
    """Calculate the attention weights.

    Q, K, V must have matching leading dimensions.
    Mask has shape broadcastable to (..., seq_len_q, seq_len_k).

    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # large negative for masking

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dm, h):
        super().__init__()
        assert dm % h == 0, "dm must be divisible by h"
        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth).
        Transpose the result to shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        K = self.Wk(K)  # (batch_size, seq_len_k, dm)
        V = self.Wv(V)  # (batch_size, seq_len_v, dm)

        Q = self.split_heads(Q, batch_size)  # (batch_size, h, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch_size, h, seq_len_k, depth)
        V = self.split_heads(V, batch_size)  # (batch_size, h, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, h, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))  # (batch_size, seq_len_q, dm)

        output = self.linear(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, attention_weights


def positional_encoding(max_len, dm):
    """Calculate positional encoding for max_len and dm dimensions"""
    pos = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    i = np.arange(dm)[np.newaxis, :]  # (1, dm)

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates

    # apply sin to even indices in the array; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(dm, hidden):
    """Point-wise feed forward network with two dense layers"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation='relu'),  # (batch_size, seq_len, hidden)
        tf.keras.layers.Dense(dm)  # (batch_size, seq_len, dm)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.ffn = point_wise_feed_forward_network(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # Self attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, dropout_rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = point_wise_feed_forward_network(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # Masked multi-head attention (self attention)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Multi-head attention (encoder-decoder attention)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab_size,
                 max_len, dropout_rate=0.1):
        super().__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, dm)
        self.pos_encoding = positional_encoding(max_len, dm)

        self.enc_layers = [
            EncoderLayer(dm, h, hidden, dropout_rate) for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # scale embeddings
        x += self.pos_encoding[:, :seq_len, :]  # add positional encoding

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, seq_len, dm)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, target_vocab_size,
                 max_len, dropout_rate=0.1):
        super().__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, dm)
        self.pos_encoding = positional_encoding(max_len, dm)

        self.dec_layers = [
            DecoderLayer(dm, h, hidden, dropout_rate) for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, input_vocab_size,
                 target_vocab_size, max_len, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab_size,
                               max_len, dropout_rate)

        self.decoder = Decoder(N, dm, h, hidden, target_vocab_size,
                               max_len, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        enc_input, dec_input, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs

        enc_output = self.encoder(enc_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, dm)

        dec_output, attention_weights = self.decoder(
            dec_input, enc_output, training, look_ahead_mask, dec_padding_mask
        )  # (batch_size, tar_seq_len, dm)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
