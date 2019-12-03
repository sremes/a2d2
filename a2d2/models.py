"""Implements image segmentation models and neural network layers."""
# pylint: disable=too-few-public-methods
import tensorflow as tf


class PositionalNormalization(tf.keras.layers.Layer):
    """Positional normalization removes the mean and standard deviation over spatial dimensions."""
    def __init__(self, epsilon=1e-5, axis=3):
        """Sets parameters of the layer.

        Args:
            epsilon: a small value to stabilize the computations
            axis: the axis along which to compute the moments (the default applies to channels in BxHxWxC)
        """
        super().__init__()
        self.epsilon = epsilon
        self.axis = axis

    def call(self, inputs):
        """

        Args:
            inputs:

        Returns:
            The input with the moments removed, and the computed mean and standard deviation that can be passed
            to a later `MomentShortcut` layer.
        """
        mean, variance = tf.nn.moments(inputs, axes=[self.axis], keepdims=True)
        standard_deviation = tf.sqrt(variance + self.epsilon)
        output = (inputs - mean) / standard_deviation
        return output, mean, standard_deviation


class MomentShortcut(tf.keras.layers.Layer):
    """The moment shortcut applies mean and standard deviation from a previous positional normalization layer."""
    def __init__(self, mean, standard_deviation):
        """Initialize the moment shortcut with given mean and standard deviation.

        Args:
            mean: the mean computed at a previous layer
            standard_deviation: the standard deviation computed at a previous layer
        """
        super().__init__()
        self.mean = mean
        self.standard_deviation = standard_deviation

    def call(self, inputs):
        """Applies the moments to the inputs.

        Args:
            inputs: a tensor with shape matching the mean and standard deviation of the layer

        Returns:
            The input with the moments applied to it
        """
        return inputs * self.standard_deviation + self.mean
