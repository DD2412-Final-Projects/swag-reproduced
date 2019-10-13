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

import utils
from networks.vgg16.vgg16 import VGG16


# Hyperparameters
LEARNING_RATE = 1e-6
EPOCHS = 10
BATCH_SIZE = 128
DISPLAY_INTERVAL = 5  # How often to display loss/accuracy during training


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
    assert args.data_path is not None, "Data path must be specified."

    return args


if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    # Load training data
    X_train = np.load(args.data_path + "X_train.npy")
    y_train = np.load(args.data_path + "y_train.npy")
    n_samples, n_classes = y_train.shape
    width, height, n_channels = X_train.shape[1:]

    # Load network architecture
    sess = tf.Session()
    X_input = tf.placeholder(tf.float32, [None, width, height, n_channels])
    y_input = tf.placeholder(tf.float32, [None, n_classes])
    vgg_network = VGG16(X_input, n_classes, args.weight_path, sess, verbose=True)
    logits = vgg_network.fc3l  # Output of the final layer

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
    train_operation = optimizer.minimize(loss)

    # Define evaluation metrics
    predictions = tf.nn.softmax(logits)
    predicted_correctly = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))

    # Initialize weights if not using pretrained weights
    if args.weight_path is None:
        init = tf.initialize_all_variables()
        sess.run(init)

    # Run training
    for epoch in range(EPOCHS):

        print("\n ---- Epoch {} ----\n".format(epoch + 1))
        X_train, y_train = utils.shuffle_data(X_train, y_train)

        for step in range(n_samples // BATCH_SIZE):

            X_batch = X_train[step: step + BATCH_SIZE]
            y_batch = y_train[step: step + BATCH_SIZE]

            sess.run(train_operation, feed_dict={X_input: X_batch, y_input: y_batch})

            if step % DISPLAY_INTERVAL == 0:
                loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X_input: X_batch, y_input: y_batch})
                print("Iteration {}, Batch loss = {}, Batch accuracy = {}".format(step + 1, loss_val, acc_val))
