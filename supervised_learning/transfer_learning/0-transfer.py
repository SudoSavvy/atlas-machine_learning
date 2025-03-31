#!/usr/bin/env python3
"""
Train a CNN to classify the CIFAR-10 dataset using Transfer Learning
"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """Preprocesses CIFAR-10 dataset"""
    X_p = X.astype("float32") / 255.0  # Normalize pixel values to [0, 1]
    Y_p = K.utils.to_categorical(Y, num_classes=10)  # One-hot encode labels
    return X_p, Y_p


def preprocess_image(image, label):
    """Resize image efficiently using a dataset pipeline"""
    image = tf.image.resize(image, (224, 224))
    return image, label


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess dataset
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Create a dataset pipeline to resize images on-the-fly
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.map(preprocess_image).batch(128).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.map(preprocess_image).batch(128).prefetch(tf.data.AUTOTUNE)

    # Load EfficientNetB0 as a feature extractor
    base_model = K.applications.EfficientNetB0(
        include_top=False,  # Remove fully connected layers
        input_shape=(224, 224, 3),
        weights="imagenet"
    )

    # Unfreeze last 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Build classification head
    inputs = K.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation="softmax")(x)

    # Create final model
    model = K.models.Model(inputs, outputs)

    # Compile with lower learning rate for fine-tuning
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model using dataset pipelines
    model.fit(train_dataset,
              validation_data=test_dataset,
              epochs=20,
              verbose=1)

    # Save model
    model.save("cifar10.keras")
