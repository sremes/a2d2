"""Tests for the `models` module."""
# pylint: disable=redefined-outer-name

import pytest
import tensorflow as tf

import a2d2.models as models


@pytest.fixture
def simple_unet_data():
    """Provide simple unet-compatible data."""
    return tf.constant(value=1.0, shape=(1, 256, 256, 1))


def test_unet_verify_output_shape(simple_unet_data):
    """Test that the output image size matches the input."""
    unet = models.UNet()
    output = unet(simple_unet_data)
    print("Input shape:", simple_unet_data.shape)
    print("Output shape:", output.shape)
    assert simple_unet_data.shape == output.shape


@pytest.mark.parametrize("number_of_classes", [1, 2, 4, 8, 16, 32, 64])
def test_number_of_classes(simple_unet_data, number_of_classes):
    """Test the number of classes in the output layer."""
    unet = models.UNet(num_classes=number_of_classes)
    output = unet(simple_unet_data)
    assert output.shape[-1] == number_of_classes
