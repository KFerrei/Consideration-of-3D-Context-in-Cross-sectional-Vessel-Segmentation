"""
File: utils/metrics/segmentation_evaluator.py
Description: Script to evaluate 2D segmentation results using Dice, Hausdorff, and Mean Contour Distance metrics.
Author: Kevin Ferreira
Date: 18 December 2024
"""

import itertools
import math
import numpy as np
from evalutils.stats import dice_from_confusion_matrix, hausdorff_distance, mean_contour_distance

class SegmentationEvaluator:
    """
    Class to evaluate 2D segmentation results using Dice coefficients, Hausdorff distance,
    and Mean Contour Distance metrics for multi-class segmentation.
    """
    def __init__(self, classes):
        """
        Initializes the evaluator with a list of valid class labels.

        Args:
            classes (list): List of class labels to evaluate.
        """
        self._classes = classes

    def validate_inputs(self, truth, prediction):
        """
        Validates the inputs for ground truth and prediction.

        Args:
            truth (np.ndarray): Ground truth segmentation array.
            prediction (np.ndarray): Predicted segmentation array.

        Raises:
            ValueError: If shapes of inputs differ or contain invalid classes.
        """
        if not truth.shape == prediction.shape:
            raise ValueError("Ground truth and prediction do not have the same size")
        if not set(truth.flatten()).issubset(self._classes):
            raise ValueError("Truth contains invalid classes")
        if not set(prediction.flatten()).issubset(self._classes):
            raise ValueError("Prediction contains invalid classes")

    def get_confusion_matrix(self, truth, prediction):
        """
        Computes the confusion matrix for multi-class segmentation.

        Args:
            truth (np.ndarray): Ground truth segmentation array.
            prediction (np.ndarray): Predicted segmentation array.

        Returns:
            np.ndarray: Confusion matrix of shape (num_classes, num_classes).
        """
        confusion_matrix = np.zeros((len(self._classes), len(self._classes)))
        for class_predicted, class_truth in itertools.product(self._classes, self._classes):
            confusion_matrix[class_truth, class_predicted] = np.sum(
                np.all(np.stack((prediction == class_predicted, truth == class_truth)), axis=0))
        return confusion_matrix

    def evaluate(self, truth: np.ndarray, prediction: np.ndarray, pixel_dim):
        """
        Evaluates the segmentation using Dice coefficients, Hausdorff distance,
        and Mean Contour Distance metrics.

        Args:
            truth (np.ndarray): Ground truth segmentation array.
            prediction (np.ndarray): Predicted segmentation array.
            pixel_dim (tuple): Pixel dimensions for distance metrics (e.g., (0.5, 0.5)).

        Returns:
            tuple: Dice coefficients, Hausdorff distances, and Mean Contour Distances as dictionaries.
        """
        self.validate_inputs(truth, prediction)
        dice_coefficients = self.evaluate_dice_coefficients(truth, prediction)
        hausdorff_distances = self.evaluate_hausdorff_distances(truth, prediction, pixel_dim)
        mean_contour_distances = self.evaluate_mean_contour_distances(truth, prediction, pixel_dim)
        return dice_coefficients, hausdorff_distances, mean_contour_distances

    def evaluate_dice_coefficients(self, truth, prediction):
        """
        Computes the Dice coefficient for each class.

        Args:
            truth (np.ndarray): Ground truth segmentation array.
            prediction (np.ndarray): Predicted segmentation array.

        Returns:
            dict: Dice coefficients for each class.
        """
        sorted_dice_coefficients = {}
        confusion_matrix = self.get_confusion_matrix(truth, prediction)
        dice_coefficients = dice_from_confusion_matrix(confusion_matrix)
        for i, class_value in enumerate(self._classes):
            sorted_dice_coefficients[class_value] = dice_coefficients[i]
        return sorted_dice_coefficients

    def evaluate_hausdorff_distances(self, truth: np.ndarray, prediction: np.ndarray, pixel_dim):
        """
        Computes the Hausdorff distance for each class.

        Args:
            truth (np.ndarray): Ground truth segmentation array.
            prediction (np.ndarray): Predicted segmentation array.
            pixel_dim (tuple): Pixel dimensions for distance metrics (e.g., (0.5, 0.5)).

        Returns:
            dict: Hausdorff distances for each class.
        """
        hausdorff_distances = {}
        for class_value in self._classes:
            try:
                hausdorff_distances[class_value] = hausdorff_distance(truth == class_value,
                                                                      prediction == class_value,
                                                                      pixel_dim)
            except ValueError:
                hausdorff_distances[class_value] = math.inf
        return hausdorff_distances

    def evaluate_mean_contour_distances(self, truth: np.ndarray, prediction: np.ndarray, pixel_dim):
        """
        Computes the Mean Contour Distance for each class.

        Args:
            truth (np.ndarray): Ground truth segmentation array.
            prediction (np.ndarray): Predicted segmentation array.
            pixel_dim (tuple): Pixel dimensions for distance metrics (e.g., (0.5, 0.5)).

        Returns:
            dict: Mean Contour Distances for each class.
        """
        mean_contour_distances = {}
        for class_value in self._classes:
            try:
                mean_contour_distances[class_value] = mean_contour_distance(truth == class_value,
                                                                            prediction == class_value,
                                                                            pixel_dim)
            except ValueError:
                mean_contour_distances[class_value] = math.inf
        return mean_contour_distances