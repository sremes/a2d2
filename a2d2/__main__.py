"""Main entrypoints for the module are defined here."""
# pylint: disable=fixme

import argparse
import os
import tensorflow as tf

import a2d2.data_loader
import a2d2.models
import a2d2.training


def parse_image_size(image_size):
    """Parse "100x100" like string into a tuple."""
    return tuple(map(int, image_size.split("x")))


def write_tf_records(args):
    """Run TFRecord writer with given arguments."""
    image_size = None
    if args.image_size is not None:
        image_size = parse_image_size(args.image_size)
    writer = a2d2.data_loader.A2D2TFRecordWriter(args.output, os.path.join(args.datadir, "class_list.json"), image_size)
    writer.write_images_to_tf_records(args.datadir)


def train_model(args):
    """Train the UNet model on TFRecords dataset."""
    image_size = parse_image_size(args.image_size)
    data_reader = a2d2.data_loader.A2D2TFRecordReader(args.tfrecord, batch_size=args.batch_size)
    model = a2d2.models.UNet(num_classes=args.classes)
    input_layer = tf.keras.Input(shape=(*image_size, 3), name="inputs")
    output_layer = model(input_layer)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)
    trainer = a2d2.training.ModelTrainer(
        model=tf.keras.Model(inputs=input_layer, outputs=output_layer), loss=loss, optimizer=optimizer
    )
    # TODO: temporary solution using as_numpy_iterator due to shape rank being unknown
    trainer.train(inputs=data_reader.get_dataset().as_numpy_iterator(), num_epochs=args.epochs)


def parse_arguments():
    """Argument parser with different sub-commands."""
    parser = argparse.ArgumentParser(description="Run various utilities and training with the A2D2 dataset")
    sub_parsers = parser.add_subparsers(required=True)

    # Handle creating TFRecords
    parser_tf_records = sub_parsers.add_parser("write_tf_records")
    parser_tf_records.add_argument("--datadir", default="camera_lidar_semantic")
    parser_tf_records.add_argument("--output", default="records")
    parser_tf_records.add_argument("--image_size", default="832x544")
    parser_tf_records.set_defaults(function=write_tf_records)

    # Handle model training
    parser_trainer = sub_parsers.add_parser("train")
    parser_trainer.add_argument("--tfrecord", default="a2d2.tfrecord")
    parser_trainer.add_argument("--image_size", default="832x544")
    parser_trainer.add_argument("--classes", default=55, type=int)
    parser_trainer.add_argument("--output", default="saved_model")
    parser_trainer.add_argument("--batch_size", default=2, type=int)
    parser_trainer.add_argument("--learning_rate", default=0.01, type=float)
    parser_trainer.add_argument("--momentum", default=0.9, type=float)
    parser_trainer.add_argument("--epochs", default=50, type=int)
    parser_trainer.set_defaults(function=train_model)

    return parser.parse_args()


def main():
    """Main function parses arguments and runs the chosen sub-command."""
    args = parse_arguments()
    args.function(args)


if __name__ == "__main__":
    main()
