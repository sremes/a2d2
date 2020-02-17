"""Test the training module."""
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import tensorflow as tf

from a2d2.training import ModelTrainer


INPUT_SHAPE = (4,)


class TrainableModel(tf.keras.Model):  # pylint: disable=too-few-public-methods
    """Simplest possible model for testing purposes."""

    def __init__(self):
        """Init a simple dense layer."""
        super().__init__()
        self.layer = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        """Apply layer to inputs."""
        return self.layer(inputs)


def model_without_input():
    """Provides a simple model for testing purposes."""
    return TrainableModel()


def compiled_model():
    """Provides an already compiled model with an input layer."""
    test_model = TrainableModel()
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name="inputs")
    model = tf.keras.Model(inputs=inputs, outputs=test_model(inputs))
    model.compile(loss=lambda x, y: 0.0)
    return model


@pytest.fixture
def inputs_and_outputs():
    """Create inputs and outputs for the test model."""
    return np.zeros((32, *INPUT_SHAPE)), np.zeros((32, 1))

@pytest.mark.parametrize("model", [model_without_input(), compiled_model()])
def test_trainer(inputs_and_outputs, model):
    """Tests the trainel class with differently initialized models."""
    loss = tf.keras.losses.MSE
    trainer = ModelTrainer(model, loss)
    trainer.train(*inputs_and_outputs)
