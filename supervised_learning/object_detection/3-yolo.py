#!/usr/bin/env python3
"""YOLO v3 Object Detection Module with NMS
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

            t_xy = 1 / (1 + np.exp(-output[..., 0:2]))
            t_wh = output[..., 2:4]
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.tile(np.arange(grid_width).reshape(1, grid_width, 1),
                         (grid_height, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_height).reshape(grid_height, 1, 1),
                         (1, grid_width, anchor_boxes))

            bx = (t_xy[..., 0] + cx) / grid_width
            by = (t_xy[..., 1] + cy) / grid_height
            bw = (self.anchors[i][:, 0] * np.exp(t_wh[..., 0])) / input_width
            bh = (self.anchors[i][:, 1] * np.exp(t_wh[..., 1])) / input_height

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

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

            scores = box_conf * box_prob
            classes = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)

            mask = scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Max Suppression to filter overlapping boxes.

        Args:
            filtered_boxes (numpy.ndarray): Array of shape (?, 4) with filtered bounding boxes.
            box_classes (numpy.ndarray): Array of shape (?,) with class numbers for each box.
            box_scores (numpy.ndarray): Array of shape (?) with box scores.

        Returns:
            Tuple of (box_predictions, predicted_box_classes, predicted_box_scores):
                box_predictions (numpy.ndarray): Filtered bounding boxes after NMS.
                predicted_box_classes (numpy.ndarray): Class predictions after NMS.
                predicted_box_scores (numpy.ndarray): Box scores after NMS.
        """
        unique_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            sorted_idxs = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[sorted_idxs]
            cls_scores = cls_scores[sorted_idxs]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                box_area = (cls_boxes[0, 2] - cls_boxes[0, 0]) * \
                           (cls_boxes[0, 3] - cls_boxes[0, 1])
                other_areas = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * \
                              (cls_boxes[1:, 3] - cls_boxes[1:, 1])

                union_area = box_area + other_areas - inter_area
                iou = inter_area / union_area

                keep = np.where(iou <= self.nms_t)

                cls_boxes = cls_boxes[1:][keep]
                cls_scores = cls_scores[1:][keep]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
