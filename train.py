"""
Trains a model using SWAG and saves the learned weight distribution parameters.

1. Load/import the network architecture from the 'networks' directory.
2. Load training data.
3. Train the model with SWAG.
4. Save the learned weight distribution parameters.
"""

import argparse
import numpy as np
import tensorflow as tf

from networks.vgg16.vgg16 import vgg16


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data that has been preprocessed using preprocess_data.py.")
    parser.add_argument("-weight_path", dest="weight_path", metavar="PATH TO WEIGTHS", default=None,
                        help="Path to pretrained weights for the network.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    # Load network architecture
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, args.weight_path, sess, verbose=True)

    # Load training data
    X_train = np.load(args.data_path + "X_train.npy")
    y_train = np.load(args.data_path + "y_train.npy")
