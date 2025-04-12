#!/usr/bin/env python3

"""0-main.py"""
import numpy as np
import tensorflow as tf
import sys

# Import yolo
sys.path.append('./')
yolo = __import__('yolo').Yolo

if __name__ == '__main__':
    model_path = 'test.h5'
    classes_path = 'test.txt'
    class_t = 0.5
    nms_t = 0.5
    anchors = np.array([[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]])

    yolo = yolo(model_path, classes_path, anchors, class_t, nms_t)

    images = np.random.randint(0, 256, (3, 512, 512, 3), dtype=np.uint8)
    pimages, image_shapes = yolo.preprocess_images(images)
    print(pimages)
    print(image_shapes)
