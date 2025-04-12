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

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs.

        Args:
            outputs (list): List of numpy.ndarrays containing predictions.
            image_size (numpy.ndarray): Original image size [height, width].

        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_height = self.model.input.shape[1]
        input_width = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Sigmoid applied to box coordinates and confidence
            t_xy = 1 / (1 + np.exp(-output[..., 0:2]))  # Sigmoid of t_x, t_y
            t_wh = output[..., 2:4]
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))  # Sigmoid of objectness score
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))  # Sigmoid of class probabilities

            # Creating grid for center offsets
            cx = np.tile(np.arange(grid_width).reshape(1, grid_width, 1), (grid_height, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_height).reshape(grid_height, 1, 1), (1, grid_width, anchor_boxes))

            # Apply center offsets and width/height
            bx = (t_xy[..., 0] + cx) / grid_width
            by = (t_xy[..., 1] + cy) / grid_height
            bw = (self.anchors[i][:, 0] * np.exp(t_wh[..., 0])) / input_width
            bh = (self.anchors[i][:, 1] * np.exp(t_wh[..., 1])) / input_height

            # Convert from center to corner coordinates
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Stack bounding boxes
            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on objectness score and class probabilities.

        Args:
            boxes (list): List of numpy.ndarrays containing processed boundary boxes.
            box_confidences (list): List of numpy.ndarrays containing box confidences.
            box_class_probs (list): List of numpy.ndarrays containing box class probabilities.

        Returns:
            Tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes (numpy.ndarray): Array of shape (?, 4) of filtered bounding boxes.
                box_classes (numpy.ndarray): Array of shape (?,) of class indices for each box.
                box_scores (numpy.ndarray): Array of shape (?) of box scores for each box.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_conf = box_confidences[i]
            box_prob = box_class_probs[i]

            # Calculate scores as objectness * class probability
            scores = box_conf * box_prob
            classes = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)

            # Filter boxes based on class_t score threshold
            mask = scores >= self.class_t

            # Collect filtered boxes, classes, and scores
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(scores[mask])

        # Concatenate results across all outputs
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
