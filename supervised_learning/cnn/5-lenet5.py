#!/usr/bin/env python3
from tensorflow import keras as K

def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.
    :param X: K.Input of shape (m, 28, 28, 1) containing the input images
    :return: a K.Model compiled with Adam optimizer and accuracy metrics
    """
    initializer = K.initializers.HeNormal(seed=0)
    
    # Convolutional Layer 1
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation='relu', kernel_initializer=initializer)(X)
    # Max Pooling Layer 1
    pool1 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)
    
    # Convolutional Layer 2
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation='relu', kernel_initializer=initializer)(pool1)
    # Max Pooling Layer 2
    pool2 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
    
    # Flatten Layer
    flat = K.layers.Flatten()(pool2)
    
    # Fully Connected Layer 1
    fc1 = K.layers.Dense(units=120, activation='relu', kernel_initializer=initializer)(flat)
    
    # Fully Connected Layer 2
    fc2 = K.layers.Dense(units=84, activation='relu', kernel_initializer=initializer)(fc1)
    
    # Output Layer with Softmax Activation
    output = K.layers.Dense(units=10, activation='softmax', kernel_initializer=initializer)(fc2)
    
    # Model Definition
    model = K.Model(inputs=X, outputs=output)
    
    # Compile Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
