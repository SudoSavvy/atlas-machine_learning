#!/usr/bin/env python3
"""
Bayesian Optimization of a Neural Network using GPyOpt
"""

import os
import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_train = x_train.reshape((-1, 28 * 28))
x_val = x_val.reshape((-1, 28 * 28))

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)


# Define the model training function
def nn_fitness(params):
    """
    Trains and evaluates a Keras model based on provided hyperparameters
    """
    lr = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2_weight = float(params[0][3])
    batch_size = int(params[0][4])

    model = Sequential([
        Dense(units, input_shape=(784,), activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(l2_weight)),
        Dropout(dropout),
        Dense(10, activation='softmax')
    ])

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    filename = f'checkpoint_lr{lr}_u{units}_do{dropout}_l2{l2_weight}_bs{batch_size}.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=0)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=0)

    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=0,
                        callbacks=[early_stop, checkpoint])

    best_val_acc = max(history.history['val_accuracy'])

    # Save best result to report
    with open('bayes_opt.txt', 'a') as f:
        f.write(f'{params[0]} -> val_accuracy: {best_val_acc:.5f}\n')

    return -best_val_acc  # Minimize negative accuracy


# Define the domain of hyperparameters
domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-1)},
    {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

# Clean previous report
if os.path.exists("bayes_opt.txt"):
    os.remove("bayes_opt.txt")

# Run Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=nn_fitness,
    domain=domain,
    acquisition_type='EI',
    exact_feval=False,
    maximize=False
)

optimizer.run_optimization(max_iter=30)

# Save convergence plot
plt.plot(-optimizer.Y)
plt.xlabel("Iteration")
plt.ylabel("Validation Accuracy")
plt.title("Bayesian Optimization Convergence")
plt.grid()
plt.savefig("convergence.png")
plt.show()
