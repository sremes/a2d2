"""Utilities for training semantic segmentation models."""
import os

import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.losses import Loss  # pylint: disable=import-error
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=import-error


class ModelTrainer:
    """Helper class for training a semantic segmentation model."""

    default_optimizer = tf.keras.optimizers.SGD(momentum=0.9)

    def __init__(self, model: Model, loss: Loss, optimizer: Optimizer = None, log_dir: str = "logs"):
        """Sets the model to be trained using this class."""
        self.model = model
        self.loss = loss
        self.optimizer = optimizer if optimizer else self.default_optimizer
        self.metrics = []
        self.callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "losses.csv")),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", verbose=1, min_delta=1e-4),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, "tensorboard")),
        ]

    def train(self, inputs, outputs, batch_size=None, num_epochs=1):
        """Train the model using the given input and output data."""
        try:
            self.model.get_layer(name="inputs")
        except ValueError:
            input_layer = tf.keras.Input(shape=inputs.shape[1:], name="inputs")
            self.model = Model(inputs=input_layer, outputs=self.model(input_layer))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model.fit(inputs, outputs, batch_size=batch_size, epochs=num_epochs, callbacks=self.callbacks)

    def predict(self, inputs, batch_size=None):
        """Make predictions using the model with given inputs."""
        return self.model.predict(inputs, batch_size=batch_size)

    def compute_loss(self, inputs, outputs, batch_size=None):
        """Compute the loss using the given inputs and outputs."""
        return self.model.evaluate(inputs, outputs, batch_size=batch_size)
