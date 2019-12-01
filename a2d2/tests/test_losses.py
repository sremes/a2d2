"""Tests for the `losses` module."""
# pylint: disable=redefined-outer-name

import pytest
import tensorflow as tf

import a2d2.losses as losses


@pytest.fixture
def simple_binary_masks():
    """Generate a simple pair of ground truth and 'predicted' masks."""
    # two simple masks
    y_true = tf.constant([[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    y_pred = tf.constant([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

    # create batch and class dimensions
    y_true = tf.expand_dims(tf.expand_dims(y_true, 0), 0)
    y_pred = tf.expand_dims(tf.expand_dims(y_pred, 0), 0)

    return y_true, y_pred


def test_giou_loss(simple_binary_masks):
    """Test the GIoU loss on an example where the true value is easy to compute by hand."""
    y_true, y_pred = simple_binary_masks
    loss = losses.GeneralizedIoULoss()
    # intersection = 3, union = 5, enclosing box area = 6
    # -> giou = 3/5 - 1/6 = 13/30 -> 1-giou = 17/30 = 0.5666...
    assert loss(y_true, y_pred).numpy() == pytest.approx(0.56666666, 0.01)


def test_find_smallest_enclosing_rectangle(simple_binary_masks):
    """Test that the enclosing rectangle is correctly found."""
    y_true, y_pred = simple_binary_masks
    loss = losses.GeneralizedIoULoss()
    rectangle = tf.squeeze(loss.find_smallest_enclosing_rectangle(y_true, y_pred)).numpy()
    assert rectangle[0] == 0
    assert rectangle[1] == 2
    assert rectangle[2] == 2
    assert rectangle[3] == 3


def test_get_enclosing_rectangle():
    """Test the helper function to find the enclosing rectangle for a single mask image."""
    along_x = tf.expand_dims(tf.expand_dims(tf.constant([0, 0, 1, 1, 0]), 0), 0)
    along_y = tf.expand_dims(tf.expand_dims(tf.constant([0, 1, 1, 1, 1]), 0), 0)

    loss = losses.GeneralizedIoULoss()
    min_x, min_y, max_x, max_y = loss.get_enclosing_rectangle(along_x, along_y)
    assert tf.squeeze(min_x).numpy() == 2
    assert tf.squeeze(max_x).numpy() == 3
    assert tf.squeeze(min_y).numpy() == 1
    assert tf.squeeze(max_y).numpy() == 4


@pytest.fixture
def nonzero_element_at_2():
    """Fixture providing sample data for testing the first element finder."""
    return tf.expand_dims(tf.expand_dims(tf.constant([0, 0, 1, 1, 0, 0, 1]), 0), 0)


def test_giou_find_first_greater_than_zero_element(nonzero_element_at_2):
    """Test that the first element is correctly found."""
    loss = losses.GeneralizedIoULoss()
    non_zero_index = loss.find_first_greater_than_zero_element(nonzero_element_at_2, axis=2)
    assert tf.squeeze(non_zero_index).numpy() == 2
