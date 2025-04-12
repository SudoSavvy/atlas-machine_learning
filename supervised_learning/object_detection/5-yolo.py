#!/usr/bin/env python3
"""
Yolo class implementation.

This module contains the Yolo class which includes methods for preprocessing images
before feeding them into a Darknet-based YOLO model.

The preprocessed images are resized to the specified input dimensions of the Darknet model
using inter-cubic interpolation. The pixel values are scaled to be in the range [0, 1].

Usage:
    yolo = Yolo(input_h, input_w)
    pimages, image_shapes = yolo.preprocess_images(images)

Where:
    - images is a list of images as numpy.ndarrays.
    - pimages is a numpy.ndarray of shape (ni, input_h, input_w, 3).
    - image_shapes is a numpy.ndarray of shape (ni, 2), where each row contains the original
      (height, width) of the corresponding image.
"""

import cv2
import numpy as np


class Yolo:
    """
    Yolo class for Darknet-based YOLO models.
    
    Attributes:
        input_h (int): The height dimension expected by the Darknet model.
        input_w (int): The width dimension expected by the Darknet model.
    """

    def __init__(self, input_h, input_w):
        """
        Initialize a Yolo instance with model-specific input dimensions.
        
        Args:
            input_h (int): Input height for the Darknet model.
            input_w (int): Input width for the Darknet model.
        """
        self.input_h = input_h
        self.input_w = input_w

    def preprocess_images(self, images):
        """
        Preprocesses images for the Darknet model.
        
        For each image in the provided list:
          - The original image shape is recorded.
          - The image is resized to (self.input_h, self.input_w) using inter-cubic interpolation.
          - The image's pixel values are scaled to the range [0, 1].

        Args:
            images (list of numpy.ndarray): List of images represented as numpy arrays.

        Returns:
            tuple:
                - pimages (numpy.ndarray): Array of preprocessed images with shape
                  (ni, input_h, input_w, 3), where ni is the number of images.
                - image_shapes (numpy.ndarray): Array of original image shapes with shape
                  (ni, 2). Each row contains (image_height, image_width).
        """
        pimages = []       # List to store preprocessed images
        image_shapes = []  # List to store original image dimensions

        for img in images:
            # Record the original image dimensions (height, width)
            image_shapes.append(img.shape[:2])
            # Resize image to (input_w, input_h) using inter-cubic interpolation
            resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
            # Rescale pixel values from [0, 255] to [0, 1] and ensure type is float32
            resized = resized.astype(np.float32) / 255.0
            pimages.append(resized)

        # Convert list of preprocessed images and image shapes to numpy arrays
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes


# Example usage (for testing purposes):
if __name__ == "__main__":
    # Create an instance of the Yolo class with model input dimensions.
    # Replace these dimensions with the appropriate values for your model.
    yolo_instance = Yolo(input_h=416, input_w=416)

    # Create dummy images for testing using numpy (each image is 480x640 with 3 color channels)
    dummy_image1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    dummy_image2 = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    images = [dummy_image1, dummy_image2]

    # Preprocess images and obtain the processed images and their original sizes
    pimages, image_shapes = yolo_instance.preprocess_images(images)

    # Output check: Displaying the shapes of the processed images and the original image sizes
    print("imagess correctly processed:", pimages.shape)
    print("image_sizes correctly calculated:", image_shapes.shape)
