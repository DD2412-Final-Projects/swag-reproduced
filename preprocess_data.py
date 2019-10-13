"""
Performs preprocessing of a dataset before training.

1. Split the dataset into train/valid/test.
2. Preprocess the input samples to the desired input format.
3. Save processed data to train, valid, and test directories.
"""

import argparse
import numpy as np
import os

import utils


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data. Default: ./data/train/")
    parser.add_argument("-train_frac", dest="train_frac", metavar="TRAIN DATA FRACTION",
                        default=0.9, help="Fraction of data to use as training data. Default: 0.9")
    parser.add_argument("-valid_frac", dest="valid_frac", metavar="VALIDATION DATA FRACTION",
                        default=0.1, help="Fraction of data to use as validation data. Default: 0.1")
    parser.add_argument("-save_path", dest="save_path", metavar="SAVE PATH", default=None,
                        help="Path to save the processed data in.")

    args = parser.parse_args()
    assert args.data_path is not None, "Path to data must be specified."
    assert os.path.exists(args.data_path), "Specified data path does not exist."
    assert args.train_frac + args.valid_frac == 1.0, "Train/valid data fractions must sum to one."
    assert args.save_path is not None, "Save path must be specified."
    if os.path.exists(args.save_path):
        response = input("Save path already exists. Previous data may be overwritten. Continue? >> (Y/n)")
        if response in ["n", "N", "no"]:
            exit()
    else:
        response = input("Save path does not exist. Create it? >> (Y/n)")
        if response in ["n", "N", "no"]:
            exit()
        os.makedirs(args.save_path)

    return args


def read_CIFAR_10(cifar_path, train=True):
    """
    Assumes the raw CIFAR-10 data is located in cifar_path,
    reads the dataset, and returns it as numpy arrays:

    data (#samples, 32, 32, 3)
    labels (#samples, 1)

    The boolean argument train determines whether the train or test set is read.
    """

    data = []
    labels = []

    if train:  # If reading train set
        for i in range(1, 6):
            file_name = cifar_path + "data_batch_" + str(i)
            data_dict = utils.unpickle(file_name)
            batch_data = data_dict[b"data"]
            batch_labels = data_dict[b"labels"]
            data.append(batch_data)
            labels.append(batch_labels)

    else:  # If reading test set
        file_name = cifar_path + "test_batch"
        data_dict = utils.unpickle(file_name)
        batch_data = data_dict[b"data"]
        batch_labels = data_dict[b"labels"]
        data.append(batch_data)
        labels.append(batch_labels)

    data = np.asarray(data)
    data = np.reshape(data, (data.shape[0] * data.shape[1], 3, 32, 32)).transpose(0, 2, 3, 1)

    labels = np.asarray(labels)
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))

    return data, labels


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # Read the dataset
    X, y = read_CIFAR_10(args.data_path, train=True)
    X_test, y_test = read_CIFAR_10(args.data_path, train=False)

    # Split into train/validation sets
    X_train, y_train, X_valid, y_valid, _, _ = utils.split_data(X, y, args.train_frac, args.valid_frac, 0, shuffle=True)

    # Preprocess the data
    # TODO

    # Save data to specified save path
    np.save(args.save_path + "X_train.npy", X_train)
    np.save(args.save_path + "y_train.npy", y_train)
    if args.valid_frac != 0:
        np.save(args.save_path + "X_valid.npy", X_valid)
        np.save(args.save_path + "y_valid.npy", y_valid)
    np.save(args.save_path + "X_test.npy", X_test)
    np.save(args.save_path + "y_test.npy", y_test)
