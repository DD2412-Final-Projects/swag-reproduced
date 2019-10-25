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
    parser.add_argument("--data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data. Default: ./data/train/")
    parser.add_argument("--train_frac", dest="train_frac", metavar="TRAIN DATA FRACTION",
                        default=0.9, help="Fraction of data to use as training data. Default: 0.9")
    parser.add_argument("--valid_frac", dest="valid_frac", metavar="VALIDATION DATA FRACTION",
                        default=0.1, help="Fraction of data to use as validation data. Default: 0.1")
    parser.add_argument("--save_path", dest="save_path", metavar="SAVE PATH", default=None,
                        help="Path to save the processed data in.")
    parser.add_argument("--data_set", dest="data_set", metavar="DATA SET", default=None,
                        help="Specify what dataset you want to preprocess (cifar10, cifar100, or stl10)")

    args = parser.parse_args()
    assert args.data_path is not None, "Path to data must be specified."
    assert os.path.exists(args.data_path), "Specified data path does not exist."
    assert args.train_frac + args.valid_frac == 1.0, "Train/valid data fractions must sum to one."
    assert args.save_path is not None, "Save path must be specified."
    assert args.data_set in ["cifar10", "cifar100", "stl10"], "Invalid choice of dataset. Must be cifar10, cifar100, or stl10."

    if os.path.exists(args.save_path):
        response = input("Save path already exists. Previous data may be overwritten. Continue? (Y/n) >> ")
        if response in ["n", "N", "no"]:
            exit()
    else:
        response = input("Save path does not exist. Create it? (Y/n) >> ")
        if response in ["n", "N", "no"]:
            exit()
        os.makedirs(os.path.join(args.save_path, ""))

    return args


def read_CIFAR_10(cifar_path, train=True):
    """
    Assumes the raw CIFAR-10 data is located in cifar_path,
    reads the dataset, and returns it as numpy arrays:

    data (#samples, 32, 32, 3)
    labels (#samples, 10)

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
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1],)).tolist()
    labels = utils.index_to_one_hot(labels, 10)

    return data, labels


def read_CIFAR_100(cifar_path, train=True):

    """

    :param cifar_path: data path for cifar-100
    :param train: check if its the train mode
    :return: data and its label

    Note:
        data (#samples, 32, 32, 3)
        labels (#samples, 100)
    """

    data = []
    labels = []

    if train:  # If reading train set

        file_name = cifar_path + "train"
        data_dict = utils.unpickle(file_name)
        batch_data = data_dict[b"data"]
        batch_labels = data_dict[b'fine_labels']
        data.append(batch_data)
        labels.append(batch_labels)

    else:  # If reading test set
        file_name = cifar_path + "test"
        data_dict = utils.unpickle(file_name)
        batch_data = data_dict[b"data"]
        batch_labels = data_dict[b"fine_labels"]
        data.append(batch_data)
        labels.append(batch_labels)

    data = np.asarray(data)
    data = np.reshape(data, (data.shape[0] * data.shape[1], 3, 32, 32)).transpose(0, 2, 3, 1)

    labels = np.asarray(labels)
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1],)).tolist()
    labels = utils.index_to_one_hot(labels, 100)

    return data, labels


def read_STL_10(stl_path, train=True):
    """
    Assumes the raw STL-10 data is located in stl_path,
    reads the dataset, and returns it as numpy arrays:

    data (#samples, 96, 96, 3)
    labels (#samples, 10)

    The boolean argument train determines whether the train or test set is read.
    """

    def read_labels(path_to_labels):
        """
        :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
        :return: an array containing the labels
        """
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return utils.index_to_one_hot(labels - 1, 10)  # convert index labels [1, 10] to one-hot

    def read_all_images(path_to_data):
        """
        :param path_to_data: the file containing the binary images from the STL-10 dataset
        :return: an array containing all the images
        """

        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.

            images = np.reshape(everything, (-1, 3, 96, 96))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    if train:
        data_path = stl_path + "train_X.bin"
        label_path = stl_path + "train_y.bin"
    else:
        data_path = stl_path + "test_X.bin"
        label_path = stl_path + "test_y.bin"

    data = read_all_images(data_path)
    labels = read_labels(label_path)

    return data, labels


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    data_path = os.path.join(args.data_path, "")
    save_path = os.path.join(args.save_path, "")

    # Read the dataset
    if args.data_set == "cifar10":
        X, y = read_CIFAR_10(data_path, train=True)
        X_test, y_test = read_CIFAR_10(data_path, train=False)
    elif args.data_set == "cifar100":
        X, y = read_CIFAR_100(data_path, train=True)
        X_test, y_test = read_CIFAR_100(data_path, train=False)
    elif args.data_set == "stl10":
        X, y = read_STL_10(data_path, train=True)
        X_test, y_test = read_STL_10(data_path, train=False)

    # Split into train/validation sets
    X_train, y_train, X_valid, y_valid, _, _ = utils.split_data(X, y, args.train_frac, args.valid_frac, 0, shuffle=True)

    # Save data to specified save path
    np.save(save_path + "X_train.npy", X_train)
    np.save(save_path + "y_train.npy", y_train)
    if args.valid_frac != 0:
        np.save(save_path + "X_valid.npy", X_valid)
        np.save(save_path + "y_valid.npy", y_valid)
    np.save(save_path + "X_test.npy", X_test)
    np.save(save_path + "y_test.npy", y_test)

    print("Processed data was saved to {} in .npy files.".format(save_path))
