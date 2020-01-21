"""Test the training module."""

import pytest
import tensorflow as tf

from a2d2.training import ModelTrainer


INPUT_SHAPE = (4,)


class TestModel(tf.keras.Model):
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
    return TestModel()


def compiled_model():
    """Provides an already compiled model with an input layer."""
    test_model = TestModel()
    inputs = tf.keras.layers.Input(shape=(INPUT_SHAPE,))
    model = tf.keras.Model(inputs=inputs, outputs=test_model(inputs))
    model.compile()
    return model


@pytest.fixture
def inputs_and_outputs():
    """Create inputs and outputs for the test model."""


@pytest.mark.parametrize("model", [model_without_input(), compiled_model()])
def test_trainer(inputs_and_outputs, model):
    loss = tf.keras.losses.MSE()
    trainer = ModelTrainer(model, loss)
    trainer.train()
