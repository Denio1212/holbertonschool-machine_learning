#!/usr/bin/env python3
"""
Initializes the Yolo Algorithm
"""


from tensorflow import keras as K


class Yolo:
    """
    Yolo class to be made
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as classes_file:
            self.class_names = [
                line.strip() for line in classes_file.readlines()
                ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
