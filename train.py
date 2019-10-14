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
import os

import utils
from networks.vgg16.vgg16 import VGG16

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Hyperparameters
tf.set_random_seed(12)
LEARNING_RATE = 5e-5
MOMENTUM = 0.9
EPOCHS = 100
BATCH_SIZE = 128
DISPLAY_INTERVAL = 1  # How often to display loss/accuracy during training


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data that has been preprocessed using preprocess_data.py.")
    parser.add_argument("--weight_path", dest="weight_path", metavar="PATH TO WEIGTHS", default=None,
                        help="Path to pretrained weights for the network.")

    parser.add_argument("--save_model", dest="save_model", metavar='SAVE MODEL PATH', default=None,
                        help="The path where the model should be saved.")

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
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input))
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
    optimizer = tf.compat.v1.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
    train_operation = optimizer.minimize(loss)

    # Define evaluation metrics
    predictions = tf.nn.softmax(logits)
    predicted_correctly = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))

    # checkpoint for getting the weights
    checkpoint = tf.train.Saver()
    save_dir = args.save_model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model')

    try:
        print("Trying to restore last checkpoint ...")

        # Use TensorFlow to find the latest checkpoint - if any.
        last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

        # Try and load the data in the checkpoint.
        checkpoint.restore(sess, save_path=last_ckpt_path)

        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_ckpt_path)
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.initialize_all_variables())
    """
    # Initialize weights if not using pretrained weights
    if args.weight_path is None:
        init = tf.initialize_all_variables()
        sess.run(init)
    """

    # Run training
    # TODO: Add SWA and SWAG
    for epoch in range(EPOCHS):

        print("\n---- Epoch {} ----\n".format(epoch + 1))
        print("Learning rate {}".format(LEARNING_RATE))
        X_train, y_train = utils.shuffle_data(X_train, y_train)
        if .9 * EPOCHS > epoch > .5 * EPOCHS:
            LEARNING_RATE -= (5e-5 - 1e-5) / .4 * EPOCHS
        for step in range(n_samples // BATCH_SIZE):

            X_batch = X_train[step: step + BATCH_SIZE]
            y_batch = y_train[step: step + BATCH_SIZE]

            sess.run(train_operation, feed_dict={X_input: X_batch, y_input: y_batch})

            if step % DISPLAY_INTERVAL == 0:
                loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X_input: X_batch, y_input: y_batch})
                print("Iteration {}, Batch loss = {}, Batch accuracy = {}".format(step + 1, loss_val, acc_val))

            # Save all variables of the TensorFlow graph to a checkpoint after each epoch.
            if (step % 50 == 0):
                checkpoint.save(sess, save_path=save_path, global_step=step)
                print("Saved checkpoint for step size:{}".format(step))
