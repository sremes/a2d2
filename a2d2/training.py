"""Utilities for training semantic segmentation models."""
import tensorflow as tf


class ModelTrainer:
    """Helper class for training a semantic segmentation model."""

    def __init__(self, model, optimizer=None):
        """Sets the model to be trained using this class."""
        self.model = model
        self.optimizer = optimizer if optimizer else tf.keras.optimizers.SGD(momentum=0.9)

    def train(self, inputs, outputs):
        """Train the model using the given input and output data."""
        input = tf.keras.Input()

    def predict(self, inputs):
        """Make predictions using the model with given inputs."""

    def loss(self, inputs, outputs):
        """Compute the loss using the given inputs and outputs."""
