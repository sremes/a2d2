"""Test the training module."""
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import tensorflow as tf

from a2d2.training import ModelTrainer


INPUT_SHAPE = (4,)
BATCH_SIZE = 32


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
    inputs = np.zeros((BATCH_SIZE, *INPUT_SHAPE))
    inputs[:, 0] = np.arange(0, BATCH_SIZE) / BATCH_SIZE
    outputs = np.arange(0, BATCH_SIZE)[:, None] / BATCH_SIZE
    return inputs, outputs


@pytest.mark.parametrize("model", [model_without_input(), compiled_model()])
def test_trainer(inputs_and_outputs, model):
    """Tests the trainel class with differently initialized models."""
    loss = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    trainer = ModelTrainer(model=model, loss=loss, optimizer=optimizer)

    trainer.train(*inputs_and_outputs, batch_size=BATCH_SIZE, num_epochs=200)

    weights = model.get_weights()
    assert (weights[0][0] - 1) ** 2 < 0.01
