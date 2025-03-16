from tensorflow.keras import layers, models

def create_model():
    """
    Create a Convolutional Neural Network (CNN) model with 5 convolutional layers
    followed by batch normalization and ReLU activation for each convolutional block.
    
    Returns:
        model (tensorflow.keras.Model): The compiled CNN model.
    """
    
    # Initialize the model as a Sequential model, meaning layers are added one after another
    model = models.Sequential()
    
    # Input layer: The model expects input of shape (224, 224, 3), typical for image data (RGB images)
    model.add(layers.InputLayer(input_shape=(224, 224, 3)))
    
    # First Convolutional Layer
    # A 2D convolution with 64 filters of size (3, 3) and padding set to 'same' to keep the output size equal to the input size.
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    
    # Batch Normalization: Helps to stabilize the learning process and improve model performance.
    model.add(layers.BatchNormalization())
    
    # ReLU Activation: Adds non-linearity to the model, helping it learn complex patterns.
    model.add(layers.ReLU())
    
    # Max Pooling: Reduces the spatial dimensions (height and width) by taking the maximum value in a 2x2 pool.
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())  # Batch normalization for stable training
    model.add(layers.ReLU())  # ReLU activation to add non-linearity

    # Third Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())  # Batch normalization for stable training
    model.add(layers.ReLU())  # ReLU activation to add non-linearity

    # Fourth Convolutional Layer
    model.add(layers.Conv2D(256, (3, 3), padding='same'))  # Increased filter size to 256

    # Fifth Convolutional Layer
    model.add(layers.Conv2D(256, (3, 3), padding='same'))  # Again, 256 filters
    model.add(layers.BatchNormalization())  # Batch normalization to maintain stable learning

    # Output Model Summary
    # Prints out the summary of the model architecture, including details of the layers and parameters
    model.summary()

    # Return the constructed model
    return model

# Create the model
model = create_model()
