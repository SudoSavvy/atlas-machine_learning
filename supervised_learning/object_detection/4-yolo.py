#!/usr/bin/env python3
"""YOLO v3 - Load Images
"""

import numpy as np
import cv2
import os


class Yolo:
    """YOLO class for object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model_path = model_path
        self.classes_path = classes_path
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a given folder

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            tuple: (images, image_paths)
                images: list of images as numpy.ndarrays
                image_paths: list of image file paths
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(file_path)

        return images, image_paths
