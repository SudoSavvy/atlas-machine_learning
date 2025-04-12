#!/usr/bin/env python3
"""YOLO v3 Preprocessing Module"""

import cv2
import numpy as np


class Yolo:
    """YOLO class for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize Yolo instance

        Args:
            model_path (str): Path to Darknet model
            classes_path (str): Path to classes file
            class_t (float): Class score threshold
            nms_t (float): Non-max suppression threshold
            anchors (numpy.ndarray): Anchor boxes
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        # Assume input dimensions are retrieved from the model itself
        # Placeholder values; in practice these would come from model input
        self.input_h = 416
        self.input_w = 416

    def preprocess_images(self, images):
        """
        Preprocess a list of images for Darknet model

        Args:
            images (list of numpy.ndarray): List of images to preprocess

        Returns:
            tuple: (pimages, image_shapes)
                pimages (numpy.ndarray): preprocessed images
                image_shapes (list of tuples): original image shapes
        """
        pimages = []
        image_shapes = []

        for img in images:
            original_shape = (img.shape[0], img.shape[1])
            image_shapes.append(original_shape)

            resized_img = cv2.resize(
                img,
                (self.input_w, self.input_h),
                interpolation=cv2.INTER_CUBIC
            )

            scaled_img = resized_img / 255.0
            pimages.append(scaled_img)

        pimages = np.array(pimages)
        # ✨ FIX: Leave image_shapes as a regular Python list (NOT np.array)
        return pimages, image_shapes
    def load_images(self, folder_path):
        """Loads and preprocesses all images in the given folder."""
        import os
        from PIL import Image
        import numpy as np

        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(path).convert('RGB')
                    img = img.resize((self.model.input_shape[1], self.model.input_shape[2]))
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    images.append(img_array)
                    image_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

        return np.array(images), image_paths
