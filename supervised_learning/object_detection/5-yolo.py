#!/usr/bin/env python3
"""
Yolo class implementation.

This module contains the Yolo class for Darknet-based YOLO models. It includes methods
to load and preprocess images before feeding them into the model.

The preprocessed images are resized to the specified input dimensions of the Darknet model
using inter-cubic interpolation, and their pixel values are scaled to the range [0, 1].

Usage:
    yolo = Yolo(model_path, classes_path, class_threshold, nms_threshold, anchors)
    images, image_paths = yolo.load_images('./yolo')
    pimages, image_shapes = yolo.preprocess_images(images)

Where:
    - images is a list of images as numpy.ndarrays.
    - image_paths is a list of file paths corresponding to the images.
    - pimages is a numpy.ndarray of shape (ni, input_h, input_w, 3), where ni is the number of images.
    - image_shapes is a numpy.ndarray of shape (ni, 2) containing the original (height, width) of each image.
"""

import os
import cv2
import numpy as np


class Yolo:
    """
    Yolo class for Darknet-based YOLO models.
    
    Attributes:
        model_path (str): Path to the Darknet model file.
        classes_path (str): Path to the file containing class names.
        class_threshold (float): Threshold for class detection scores.
        nms_threshold (float): Threshold for non-max suppression.
        anchors (list): List of anchor boxes.
        input_h (int): The height dimension expected by the Darknet model.
        input_w (int): The width dimension expected by the Darknet model.
    """

    def __init__(self, model_path, classes_path, class_threshold, nms_threshold, anchors):
        """
        Initialize a Yolo instance with the provided parameters.
        
        Default input dimensions (input_h and input_w) are set to 416.
        These can be modified if a different Darknet model input size is required.
        
        Args:
            model_path (str): Path to the Darknet model file.
            classes_path (str): Path to the file containing class names.
            class_threshold (float): Threshold for class detection scores.
            nms_threshold (float): Threshold for non-max suppression.
            anchors (list): List of anchor boxes.
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.class_threshold = class_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors

        # Set default input dimensions for the model (can be modified)
        self.input_h = 416
        self.input_w = 416

    def load_images(self, folder_path):
        """
        Loads images from the specified folder.
        
        This method iterates through the given folder and loads each image file
        whose extension indicates a common image format. It returns a tuple:
            - images: List of images loaded as numpy.ndarrays.
            - image_paths: List of corresponding image file paths.
        
        Args:
            folder_path (str): Path to the folder containing image files.
            
        Returns:
            tuple: (images, image_paths)
                - images (list of numpy.ndarray): The loaded images.
                - image_paths (list of str): The file paths for the loaded images.
        """
        images = []
        image_paths = []
        # Define supported image extensions
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # List files in the folder in sorted order for consistent behavior
        for file in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_ext:
                full_path = os.path.join(folder_path, file)
                img = cv2.imread(full_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(full_path)
        return images, image_paths

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
            # Rescale pixel values from [0, 255] to [0, 1] and convert to float32
            resized = resized.astype(np.float32) / 255.0
            pimages.append(resized)

        # Convert lists to numpy arrays
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes


# Example usage (for testing purposes):
if __name__ == "__main__":
    # Define dummy anchors for testing purposes.
    anchors = [[10, 13, 16, 30, 33, 23]]
    
    # Create an instance of the Yolo class.
    # This matches the expected parameters: model_path, classes_path, class_threshold, nms_threshold, anchors.
    yolo_instance = Yolo('test.h5', 'test.txt', 0.6, 0.5, anchors)

    # Load images from the './yolo' directory.
    # This directory should contain image files with a supported extension.
    images, image_paths = yolo_instance.load_images('./yolo')

    # Preprocess the loaded images.
    pimages, image_shapes = yolo_instance.preprocess_images(images)

    # Output check: Display the shapes of the processed images and the original image sizes.
    print("imagess correctly processed:", pimages.shape)
    print("image_sizes correctly calculated:", image_shapes.shape)
