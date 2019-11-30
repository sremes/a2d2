"""Main entrypoints for the module are defined here."""

import argparse
import os

import a2d2.data_loader


def write_tf_records(args):
    """Run TFRecord writer with given arguments."""
    writer = a2d2.data_loader.A2D2TFRecordWriter(args.output, os.path.join(args.datadir, "class_list.json"))
    writer.write_images_to_tf_records(args.datadir)


def parse_arguments():
    """Argument parser with different sub-commands."""
    parser = argparse.ArgumentParser(description="Run various utilities and training with the A2D2 dataset")
    sub_parsers = parser.add_subparsers(required=True)

    # Handle creating TFRecords
    parser_tf_records = sub_parsers.add_parser("write_tf_records")
    parser_tf_records.add_argument("--datadir", default="camera_lidar_semantic")
    parser_tf_records.add_argument("--output", default="records")
    parser_tf_records.set_defaults(function=write_tf_records)

    return parser.parse_args()


def main():
    """Main function parses arguments and runs the chosen sub-command."""
    args = parse_arguments()
    args.function(args)


if __name__ == "__main__":
    main()
