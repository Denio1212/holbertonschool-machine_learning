#!/usr/bin/env python3
"""
Initializes the Yolo Algorithm
"""


from tensorflow import keras as K
import numpy as np
import os
import cv2


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

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the Yolo model

        :param outputs: the outputs of the model

        :param image_size: the size of the image
        """
        image_height, image_height = image_size

        boxes = []
        box_confidences = []
        box_class_probs = []

        for idx, output in enumerate(outputs):
            grid_height, grid_width, nbr_anchor, _ = output.shape

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))

            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            p_w = self.anchors[idx, :, 0]
            p_h = self.anchors[idx, :, 1]

            image_height, image_width = image_size

            b_x = ((1.0 / (1.0 + np.exp(-t_x))) + grid_x) / grid_width
            b_y = ((1.0 / (1.0 + np.exp(-t_y))) + grid_y) / grid_height

            b_w = p_w * np.exp(t_w)
            b_w /= self.model.input.shape[1]
            b_h = p_h * np.exp(t_h)
            b_h /= self.model.input.shape[2]

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_w / 2 + b_x) * image_width
            y2 = (b_h / 2 + b_y) * image_height

            box = np.zeros((grid_height, grid_width, nbr_anchor, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            confidences = output[:, :, :, 4:5]
            sigmoid_confidence = 1 / (1 + np.exp(-confidences))
            class_probs = output[:, :, :, 5:]
            sigmoid_class_probs = 1 / (1 + np.exp(-class_probs))

            box_confidences.append(sigmoid_confidence)
            box_class_probs.append(sigmoid_class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
            Public method to filter boxes
        """

        filtered_boxes = np.empty((0, 4))
        box_classes = np.empty((0,), dtype=int)
        box_scores = np.empty(0, dtype=int)

        for i in range(len(boxes)):
            box_score = np.multiply(box_confidences[i], box_class_probs[i])

            box_classes_i = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            filtering_mask = box_class_score >= self.class_t

            filtered_boxes = np.concatenate((filtered_boxes,
                                             boxes[i][filtering_mask]), axis=0)
            box_classes = (
                np.concatenate((box_classes,
                                box_classes_i[filtering_mask]),
                               axis=0))
            box_scores = np.concatenate((box_scores,
                                         box_class_score[filtering_mask]),
                                        axis=0)

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, box2):
        """
            Execute Intersection Over Union
        """
        b1x1, b1y1, b1x2, b1y2 = tuple(box1)
        b2x1, b2y1, b2x2, b2y2 = tuple(box2)

        x1 = np.maximum(b1x1, b2x1)
        y1 = np.maximum(b1y1, b2y1)
        x2 = np.minimum(b1x2, b2x2)
        y2 = np.minimum(b1y2, b2y2)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        union = area1 + area2 - intersection

        result = intersection / union

        return result

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-maximum suppression

        :param filtered_boxes: filtered boxes done in the above function

        :param box_classes: the classes applied to the boxes

        :param box_scores: the scores applied to the boxes
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)
        for cls in unique_classes:

            class_indices = np.where(box_classes == cls)[0]

            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            while len(class_boxes) > 0:
                max_score_index = np.argmax(class_scores)

                box_predictions.append(class_boxes[max_score_index])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_scores[max_score_index])

                ious = np.array([self.iou(class_boxes[max_score_index],
                                          box) for box in class_boxes])

                above_threshold = np.where(ious > self.nms_t)[0]

                if len(class_boxes) > 0:
                    class_boxes = np.delete(class_boxes, above_threshold,
                                            axis=0)
                    class_scores = np.delete(class_scores, above_threshold)

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from folder

        :param folder_path: The given PATH

        :returns: images -> a list of images as numpy arrays
        image_paths -> a list of the image paths
        """
        images = []
        path = []

        for find in os.listdir(folder_path):
            if find.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                images_path = os.path.join(folder_path, find)
                image = cv2.imread(images_path)

            if image is not None:
                images.append(image)
                path.append(images_path)

        return images, path

    def preprocess_images(self, images):
        """
        Preprocesses images before feeding into the network

        :images: a list of images as numpy arrays

        Resizes the images with inter-cubic interpolation

        rescales the images to values in range [0, 1]

        :returns:
        pimages -> numpy array of shape (ni, input_h, input_w, 3)
        containing all the preprocessed images

        image_shapes -> numpy array of shape (ni, 2) containing
        the original image shape
        """
        pimages = []
        image_shapes = []

        for image in images:
            h, w, c = image.shape
            image_shapes.append([h, w])

            input_h = self.model.input.shape[1]
            input_w = self.model.input.shape[2]
            resized_img = cv2.resize(image,
                                     dsize=(
                                         input_h,
                                         input_w),
                                     interpolation=cv2.INTER_CUBIC)

            resized_img = resized_img / 255.0

            pimages.append(resized_img)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
