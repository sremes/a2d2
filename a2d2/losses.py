"""The `losses` module implements loss functions used for semantic segmentation."""

import tensorflow as tf


class GeneralizedIoULoss(tf.keras.losses.Loss):
    """Implements the generalized GIoU loss for semantic segmentation.

    For reference, see: https://arxiv.org/abs/1902.09630
    """

    def call(self, y_true, y_pred):
        """Computes the generalized IoU loss.

        Args:
            y_true: ground truth binary masks with shape (batches, classes, width, height)
            y_pred: predicted binary masks with the same shape as above

        Returns:
            The value of the loss.
        """
        # Compute IoU
        true_positives, false_positives, false_negatives = self._get_confusion_matrix(y_true, y_pred)
        intersection = tf.cast(tf.reduce_sum(true_positives), tf.float32)
        union = tf.cast(tf.reduce_sum(true_positives + false_positives + false_negatives), tf.float32)
        intersection_over_union = intersection / union
        # Compute the enclosing area
        enclosing_box = self._find_smallest_enclosing_rectangle(y_true, y_pred)
        enclosing_area = self._compute_box_area(enclosing_box)
        # Compute the loss as `1 - GIoU`
        enclosing_ratio = (enclosing_area - union) / enclosing_area
        giou = intersection_over_union - enclosing_ratio
        return 1 - giou

    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Computes the confusion matrix between the ground truth and predictions.

        Args:
            y_true: ground truth binary masks with shape (batches, classes, width, height)
            y_pred: predicted binary masks with the same shape as above

        Returns:
            A tuple containing the true positives, false positives and false negatives.
        """
        true_positives = y_pred * y_true
        false_positives = y_pred * (1 - y_true)
        false_negatives = (1 - y_pred) * y_true
        return (true_positives, false_positives, false_negatives)

    @staticmethod
    def find_smallest_enclosing_rectangle(y_true, y_pred):
        """Finds the smallest rectangle that encloses the areas of `y_true` and `y_pred`.

        Args:
            y_true: ground truth binary masks with shape (batches, classes, width, height)
            y_pred: predicted binary masks with the same shape as above

        Returns:
            A tensor containing the corners of the enclosing rectangle
        """
        # get rectangle for the predictions mask
        predictions_along_x = tf.reduce_sum(y_pred, axis=3)
        predictions_along_y = tf.reduce_sum(y_pred, axis=2)
        predictions_rectangle = GeneralizedIoULoss.get_enclosing_rectangle(predictions_along_x, predictions_along_y)

        # get rectangle for the ground truth mask
        true_along_x = tf.reduce_sum(y_true, axis=3)
        true_along_y = tf.reduce_sum(y_true, axis=2)
        true_rectangle = GeneralizedIoULoss.get_enclosing_rectangle(true_along_x, true_along_y)

        x_0 = tf.minimum(predictions_rectangle[0], true_rectangle[0])
        y_0 = tf.minimum(predictions_rectangle[1], true_rectangle[1])
        x_1 = tf.maximum(predictions_rectangle[2], true_rectangle[2])
        y_1 = tf.maximum(predictions_rectangle[3], true_rectangle[3])

        return tf.stack([x_0, y_0, x_1, y_1], axis=-1)

    @staticmethod
    def get_enclosing_rectangle(predictions_along_x, predictions_along_y, axis=2):
        """Finds the enclosing rectangle from given x and y marginals."""
        min_x = GeneralizedIoULoss.find_first_greater_than_zero_element(predictions_along_x, axis=axis)
        min_y = GeneralizedIoULoss.find_first_greater_than_zero_element(predictions_along_y, axis=axis)

        max_x = GeneralizedIoULoss.find_first_greater_than_zero_element(
            tf.reverse(predictions_along_x, [axis]), axis=axis
        )
        max_x = tf.shape(predictions_along_x, out_type=tf.int64)[axis] - max_x - 1  # count from the beginning

        max_y = GeneralizedIoULoss.find_first_greater_than_zero_element(
            tf.reverse(predictions_along_y, [axis]), axis=axis
        )
        max_y = tf.shape(predictions_along_y, out_type=tf.int64)[axis] - max_y - 1  # count from the beginning

        return (min_x, min_y, max_x, max_y)

    @staticmethod
    def find_first_greater_than_zero_element(tensor, axis=2):
        """Finds the first element greater than zero on a given `axis`.

        Args:
            tensor: tensor where to find the elements
            axis: the axis along which to search

        Returns:
            A tensor with the indices where the first element was found.
        """
        return tf.argmax(tf.cast(tf.greater(tensor, 0), tf.int64), axis=axis)

    @staticmethod
    def compute_box_area(box):
        """Compute are of a box defined by x0, y0, x1, y1 indices.

        Args:
            box: tensor with shape (batch, class, x0y0x1y2)

        Returns:
            Areas of the boxes
        """
        return tf.cast((1 + box[:, :, 2] - box[:, :, 0]) * (1 + box[:, :, 3] - box[:, :, 1]), tf.float32)
