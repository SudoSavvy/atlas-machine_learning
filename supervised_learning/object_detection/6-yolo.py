#!/usr/bin/env python3
"""
Yolo class implementation.

This module contains the Yolo class for Darknet-based YOLO models. It includes methods
to load images, preprocess images, and display images with boundary boxes, class names, and
box scores.

The preprocessed images are resized to the specified input dimensions of the Darknet model
using inter-cubic interpolation and their pixel values are scaled to the range [0, 1].

Usage:
    yolo = Yolo(model_path, classes_path, class_threshold, nms_threshold, anchors)
    images, image_paths = yolo.load_images('./yolo')
    pimages, image_shapes = yolo.preprocess_images(images)
    yolo.show_boxes(image, boxes, box_classes, box_scores, file_name)

Where:
    - images is a list of images as numpy.ndarrays.
    - image_paths is a list of file paths corresponding to the images.
    - pimages is a numpy.ndarray of shape (ni, input_h, input_w, 3) where ni is the number of images.
    - image_shapes is a numpy.ndarray of shape (ni, 2) containing the original (height, width) of each image.
    - file_name is the original file path of the image, which will be used as the window name and saved file name.
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
        class_names (list): List of class names loaded from classes_path.
        input_h (int): The height dimension expected by the Darknet model.
        input_w (int): The width dimension expected by the Darknet model.
    """

    def __init__(self, model_path, classes_path, class_threshold, nms_threshold, anchors):
        """
        Initialize a Yolo instance with the provided parameters.
        
        The input dimensions (input_h and input_w) are set to default values (416).
        These may be modified if a different Darknet model input size is required.
        Also, the class names are loaded from the given classes_path.
        
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

        # Set default input dimensions for the model
        self.input_h = 416
        self.input_w = 416

        # Load class names from the classes file, one class per line
        with open(self.classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]

    def load_images(self, folder_path):
        """
        Loads images from the specified folder.
        
        Iterates through the folder and loads each image with a valid extension.
        Returns:
            tuple: (images, image_paths)
                - images (list of numpy.ndarray): Loaded images.
                - image_paths (list of str): File paths of the loaded images.
        """
        images = []
        image_paths = []
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
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
        
        For each image:
          - Records the original image dimensions.
          - Resizes the image to (input_h, input_w) with inter-cubic interpolation.
          - Scales pixel values from [0, 255] to [0, 1].
        
        Args:
            images (list of numpy.ndarray): List of images.
            
        Returns:
            tuple:
                - pimages (numpy.ndarray): Array of preprocessed images of shape
                  (ni, input_h, input_w, 3) where ni is the number of images.
                - image_shapes (numpy.ndarray): Array of original image sizes (height, width)
                  with shape (ni, 2).
        """
        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
            resized = resized.astype(np.float32) / 255.0
            pimages.append(resized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with boundary boxes, class names, and box scores.
        
        The method draws each box with a blue line of thickness 2. Above each box,
        the class name and box score (rounded to 2 decimal places) are drawn in red,
        positioned 5 pixels above the top-left corner using FONT_HERSHEY_SIMPLEX, font
        scale 0.5, line thickness 1, and LINE_AA for line type.
        
        The window is titled with file_name. If the 's' key is pressed, the image is
        saved to a 'detections' directory (created if necessary) with the same file name.
        Any other key closes the window without saving.
        
        Args:
            image (numpy.ndarray): The original unprocessed image.
            boxes (numpy.ndarray): Array containing boundary boxes (each box as [x1, y1, x2, y2]).
            box_classes (numpy.ndarray): Array with the class indices for each box.
            box_scores (numpy.ndarray): Array with the scores for each box.
            file_name (str): File path where the original image is stored. Used for window title and saved image name.
        """
        # Create a copy of the image so the original is not modified
        img_display = image.copy()

        # Loop through all boxes to draw them and write text above the box
        for box, class_idx, score in zip(boxes, box_classes, box_scores):
            # Draw the box: blue color in BGR and thickness 2
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(img_display, top_left, bottom_right, color=(255, 0, 0), thickness=2)
            
            # Prepare the text (class name and score rounded to 2 decimal places)
            # Ensure that class_idx is a valid index in self.class_names
            if class_idx < len(self.class_names):
                label = f"{self.class_names[class_idx]} {score:.2f}"
            else:
                label = f"{class_idx} {score:.2f}"
            
            # Determine the position for the text: 5 pixels above the top-left corner of the box
            text_position = (top_left[0], max(top_left[1] - 5, 0))
            cv2.putText(
                img_display, label, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA
            )

        # Display the image in a window with the name set to file_name
        cv2.imshow(file_name, img_display)
        key = cv2.waitKey(0) & 0xFF

        # If the key pressed is 's', save the image in the 'detections' directory
        if key == ord('s'):
            # Create detections directory if it does not exist
            os.makedirs('detections', exist_ok=True)
            # Save the image using the same file name as provided (only the basename)
            save_path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(save_path, img_display)

        # Close the image window
        cv2.destroyAllWindows()


# Example usage (for testing purposes):
if __name__ == "__main__":
    # Define dummy anchors for testing purposes.
    anchors = [[10, 13, 16, 30, 33, 23]]
    
    # Create an instance of the Yolo class.
    yolo_instance = Yolo('test.h5', 'test.txt', 0.6, 0.5, anchors)

    # Load images from the './yolo' directory.
    images, image_paths = yolo_instance.load_images('./yolo')
    
    # Preprocess the loaded images.
    pimages, image_shapes = yolo_instance.preprocess_images(images)
    print("imagess correctly processed:", pimages.shape)
    print("image_sizes correctly calculated:", image_shapes.shape)
    
    # For demonstration, assume a test image with dummy boxes, class indices, and scores.
    if images:
        test_image = images[0]
        # Dummy boxes format: [x1, y1, x2, y2]
        boxes = np.array([
            [50, 50, 200, 200],
            [150, 80, 300, 230]
        ])
        # Dummy class indices (assumes at least 2 classes are defined)
        box_classes = np.array([0, 1])
        # Dummy box scores
        box_scores = np.array([0.85, 0.76])
        # Use the first image's file name for testing
        file_name = image_paths[0] if image_paths else 'test.jpg'
        
        # Display the image with boxes using the show_boxes method.
        yolo_instance.show_boxes(test_image, boxes, box_classes, box_scores, file_name)
