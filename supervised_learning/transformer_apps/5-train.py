#!/usr/bin/env python3
"""
Train a Transformer model on Portuguese to English translation.
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = scce(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.equal(y_true, y_pred)

    mask = tf.not_equal(y_true, 0)
    matches = tf.logical_and(mask, matches)

    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    # Load dataset
    data = Dataset(batch_size, max_len)

    # Create Transformer
    transformer = Transformer(N, dm, h, hidden, max_len,
                              len(data.tokenizer_pt),
                              len(data.tokenizer_en))

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(pt, en):
        en_input = en[:, :-1]
        en_real = en[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(pt, en_input)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(pt, en_input, True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(en_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        acc = accuracy_function(en_real, predictions)

        train_loss(loss)
        train_accuracy(acc)

    for epoch in range(1, epochs + 1):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch_num, (pt, en) in enumerate(data.data_train):
            train_step(pt, en)

            if batch_num % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_num}: Loss {train_loss.result():.6f} Accuracy {train_accuracy.result():.6f}")

        print(f"Epoch {epoch}: Loss {train_loss.result():.6f} Accuracy {train_accuracy.result():.6f}")

    return transformer
