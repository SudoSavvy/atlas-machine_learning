class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, max_len, input_vocab, target_vocab):
        # initialize your transformer architecture here
        # N blocks, dm dimensionality, h heads, etc.

    def call(self, inputs, targets, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # define forward pass, return predictions, attention weights etc.
        return output, attention_weights
