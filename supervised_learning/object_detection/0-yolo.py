#!/usr/bin/env python3
"""YOLO v3 Object Detection Module
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class for performing object detection using YOLO v3."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo class.

        Args:
            model_path (str): Path to Darknet Keras model.
            classes_path (str): Path to text file containing class names.
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array containing all anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """
        Load class names from a file.

        Args:
            classes_path (str): Path to file containing class names.

        Returns:
            List of class names.
        """
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
