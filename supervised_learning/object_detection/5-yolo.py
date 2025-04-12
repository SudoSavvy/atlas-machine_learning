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
                image_shapes (numpy.ndarray): original image shapes
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
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
