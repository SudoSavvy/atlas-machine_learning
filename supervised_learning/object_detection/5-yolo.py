#!/usr/bin/env python3
"""
YOLO class
"""

import numpy as np
import cv2
import os


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model_path = model_path
        self.classes_path = classes_path
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        # Load model
        self.model = self.load_model(model_path)
        
        # Load classes
        self.class_names = self.load_classes(classes_path)

    def load_model(self, model_path):
        # Load Keras model
        from tensorflow.keras.models import load_model
        return load_model(model_path)

    def load_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def load_images(self, folder_path):
        """Load images from a folder"""
        image_paths = []
        images = []

        # Walk through folder and collect image paths
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(folder_path, filename)
                image_paths.append(path)

        # Read and normalize images
        for path in image_paths:
            img = cv2.imread(path)
            img = img.astype('float32') / 255.0  # Normalize pixel values
            images.append(img)

        return images, image_paths
