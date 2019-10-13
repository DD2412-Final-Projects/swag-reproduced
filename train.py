"""
Trains a model using SWAG and saves the learned weight distribution parameters.

1. Load/import the network architecture from the 'networks' directory.
2. Load training data.
3. Train the model with SWAG.
4. Save the learned weight distribution parameters.
"""

import argparse


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", dest="train_data_path", metavar="PATH TO TRAINING DATA", default="./data/train/",
                        help="Path to training data. Default: ./data/train/")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    # Load training data
    train_data_path = args.train_data_path
