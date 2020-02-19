"""Utilities to deal with the A2D2 dataset."""

import json
import os
import re

import numpy as np
from PIL import Image, ImageColor
import tensorflow as tf


class A2D2TFRecordWriter:
    """Write TFRecords from the raw A2D2 dataset."""

    # Define features used within the tf.Examples
    feature_types = {
        "image": lambda image: tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "shape": lambda shape: tf.train.Feature(int64_list=tf.train.Int64List(value=[*shape])),
    }

    def __init__(self, tf_record_file, class_list_path, image_size=None):
        """Sets target directory for the TFRecords and reads the class definitions.

        Args:
            tf_record_file: path where to save the TFRecord files
            class_list_path: path to a json file giving the class descriptions of the labels
            image_size: the target image size; no resizing if None
        """
        self.tf_record_file = tf_record_file
        with open(class_list_path, "r") as file:
            self.class_list = json.load(file)
        self.image_size = image_size

    def write_images_to_tf_records(self, image_directory):
        """Writes images from a directory into TFRecords.

        Args:
            image_directory: path to the directory containing the data
        """
        with tf.io.TFRecordWriter(self.tf_record_file) as writer:
            for record in self.find_images_from_directory(image_directory):
                example = self.serialize_record(record)
                writer.write(example)

    def serialize_record(self, record_paths):
        """Serialize the given image and label mask into a `tf.train.Example`."""
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    **self.serialize_image(record_paths["image"]),
                    **self.serialize_label_masks(record_paths["label_masks"]),
                }
            )
        ).SerializeToString()

    def serialize_image(self, image_path):
        """Serialize given image."""
        image = Image.open(image_path, "r")
        if self.image_size is not None:
            image = image.resize(self.image_size)
        image = np.array(image)
        return {
            "image/data": self.feature_types["image"](image.tobytes()),
            "image/shape": self.feature_types["shape"](image.shape),
        }

    def serialize_label_masks(self, labels_path):
        """Serialize label masks.

        The given input masks are color coded, with one color corresponding to one class. This method splits the
        classes into separate binary masks.

        Args:
              labels_path: path to the label image

        Returns `dict` containing the image data and its shape.
        """
        image = Image.open(labels_path, "r")
        if self.image_size is not None:
            image = image.resize(self.image_size)
        masks = []
        for color_code, _ in self.class_list.items():
            color = ImageColor.getrgb(color_code)
            masks.append(np.all(np.array(image) == color, axis=-1))
        masks = np.stack(masks, axis=-1)
        return {
            "label_masks/data": self.feature_types["image"](masks.tobytes()),
            "label_masks/shape": self.feature_types["shape"](masks.shape),
        }

    @staticmethod
    def find_images_from_directory(directory):
        """Find images from a nested directory structure.

        The images are found in a path relative to `directory` as
            "YYYYMMDD_HHMMSS/camera/cam_front_center/YYYYMMDDHHMMSS_camera_frontcenter_xxxxxxxxx.png"
        with a corresponding label file stored in
            "YYYYMMDD_HHMMSS/label/cam_front_center/YYYYMMDDHHMMSS_label_frontcenter_xxxxxxxxx.png".

        Args:
            directory: path to the root of the directory structure

        Returns a generator yielding tuples of image and label pairs.
        """
        sub_directories = [
            os.path.join(directory, listed_file)
            for listed_file in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, listed_file))
        ]
        for sub_directory in sub_directories:
            identifier = sub_directory.rsplit("/", 1)[-1].replace("_", "")
            image_directory = os.path.join(sub_directory, "camera/cam_front_center")
            image_pattern = f"{identifier}_camera_frontcenter_" + r"[0-9]+\.png"
            labels_directory = os.path.join(sub_directory, "label/cam_front_center")
            labels_pattern = f"{identifier}_label_frontcenter_" + r"[0-9]+\.png"
            images = sorted([f for f in os.listdir(image_directory) if re.match(image_pattern, f)])
            labels = sorted([f for f in os.listdir(labels_directory) if re.match(labels_pattern, f)])
            for image_name, labels_name in zip(images, labels):
                yield {
                    "image": os.path.join(image_directory, image_name),
                    "label_masks": os.path.join(labels_directory, labels_name),
                }
