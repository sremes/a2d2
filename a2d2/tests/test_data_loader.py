"""Tests for the `data_loader` module."""

import os
import pytest

import a2d2.data_loader as data_loader


@pytest.mark.skipif(not os.path.exists("a2d2.tfrecord"), reason="needs access to a tfrecord")
def test_simple():
    """Tests loading a local tfrecord file."""
    batch_size = 16
    reader = data_loader.A2D2TFRecordReader("a2d2.tfrecord", batch_size)
    dataset = reader.get_dataset()
    images, labels = next(dataset.as_numpy_iterator())
    print(images.shape)
    print(labels.shape)
    assert images.shape[0] == batch_size
    assert labels.shape[0] == batch_size
