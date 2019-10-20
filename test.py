"""
Reads learned weight distribution parameters from file,
and tests using the SWAG test procedure.

1. Load the learned parameters.
2. Load test data.
3. Run SWAG test procedure for each sample.
4. Compute metrics and generate plots (e.g. NLL, reliability diagrams).
"""
import argparse
import numpy as np
import tensorflow as tf
import os

import utils
from networks.vgg16.vgg16 import VGG16

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

BATCH_SIZE = 128


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data that has been preprocessed using preprocess_data.py.")
    parser.add_argument("--load_weight_file", dest="load_weight_file", metavar="LOAD WEIGHT FILE", default=None,
                        help="File to load trained weights for the network from.")
    parser.add_argument("--load_checkpoint_path", dest="load_checkpoint_path", metavar='LOAD CHECKPOINT PATH', default=None,
                        help="Path to load checkpoint from.")

    args = parser.parse_args()
    assert args.data_path is not None, "Data path must be specified."
    assert args.load_weight_file is not None or args.load_checkpoint_path is not None, "Weight file or checkpoint path must be specified."

    return args


if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    # Load training data
    X_test = np.load(args.data_path + "X_test.npy")
    y_test = np.load(args.data_path + "y_test.npy")
    n_samples, n_classes = y_test.shape
    width, height, n_channels = X_test.shape[1:]

    # Load network architecture
    sess = tf.Session()
    X_input = tf.placeholder(tf.float32, [None, width, height, n_channels])
    y_input = tf.placeholder(tf.float32, [None, n_classes])
    vgg_network = VGG16(X_input, n_classes, weights=args.load_weight_file, sess=sess)
    logits = vgg_network.fc3l  # Output of the final layer

    # Define evaluation metrics
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input))
    predictions = tf.nn.softmax(logits)
    predicted_correctly = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    loss_test, acc_test = [], []

    if args.load_checkpoint_path is not None:

        try:
            print("Trying to restore last checkpoint ...")
            checkpoint = tf.train.Saver()

            # Use TensorFlow to find the latest checkpoint - if any.
            last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=args.load_checkpoint_path)
            print(last_ckpt_path)
            # Try and load the data in the checkpoint.
            checkpoint.restore(sess, save_path=last_ckpt_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_ckpt_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint.")

    print("Computing test performance..")
    for step in range(n_samples // BATCH_SIZE):

        X_batch = X_test[step: step + BATCH_SIZE]
        y_batch = y_test[step: step + BATCH_SIZE]

        loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict={X_input: X_batch, y_input: y_batch})
        loss_test.append(loss_batch)
        acc_test.append(acc_batch)

    print('Loss: {} \nAccuracy: {}'.format(np.mean(loss_test), np.mean(acc_test)))
