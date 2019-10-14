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
LEARNING_RATE = 5e-10
MOMENTUM = 0.9
EPOCHS = 10
BATCH_SIZE = 128
DISPLAY_INTERVAL = 10  # How often to display loss/accuracy during training (steps)
CHECKPOINT_INTERVAL = 10  # How often to save checkpoints (epochs)


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data that has been preprocessed using preprocess_data.py.")
    parser.add_argument("--save_weight_path", dest="save_weight_path", metavar="SAVE WEIGHT PATH", default=None,
                        help="Path to save trained weights for the network to.")
    parser.add_argument("--save_checkpoint_path", dest="save_checkpoint_path", metavar='SAVE CHECKPOINT PATH', default=None,
                        help="Path to save checkpoints to.")
    parser.add_argument("--load_checkpoint_path", dest="load_checkpoint_path", metavar='LOAD CHECKPOINT PATH', default=None,
                        help="Path to load checkpoint from.")

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
    vgg_network = VGG16(X_input, n_classes, weights=None, sess=sess)
    logits = vgg_network.fc3l  # Output of the final layer

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
    # optimizer = tf.compat.v1.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
    train_operation = optimizer.minimize(loss)

    # Define evaluation metrics
    predictions = tf.nn.softmax(logits)
    predicted_correctly = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))

    # Create path for checkpoints
    if args.save_checkpoint_path is not None:
        checkpoint = tf.train.Saver()
        save_dir = args.save_checkpoint_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'model')

    if args.load_checkpoint_path is not None:
        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=args.load_checkpoint_path)

            # Try and load the data in the checkpoint.
            checkpoint.restore(sess, save_path=last_ckpt_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_ckpt_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. Initializing variables instead.")
            sess.run(tf.initialize_all_variables())
    else:
        sess.run(tf.initialize_all_variables())

    # Run training
    for epoch in range(EPOCHS):

        X_train, y_train = utils.shuffle_data(X_train, y_train)

        print("\n---- Epoch {} ----\n".format(epoch + 1))
        print("Learning rate {}".format(LEARNING_RATE))
        if .9 * EPOCHS >= epoch >= .5 * EPOCHS:
            LEARNING_RATE -= (5e-10 - 1e-10) / (.4 * EPOCHS)  # Linear decay from 5e-10 to 1e-10 over 40% of epochs

        for step in range(n_samples // BATCH_SIZE):

            X_batch = X_train[step: step + BATCH_SIZE]
            y_batch = y_train[step: step + BATCH_SIZE]

            sess.run(train_operation, feed_dict={X_input: X_batch, y_input: y_batch})

            if step % DISPLAY_INTERVAL == 0:
                loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X_input: X_batch, y_input: y_batch})
                print("Iteration {}, Batch loss = {}, Batch accuracy = {}".format(step + 1, loss_val, acc_val))

        # Save all variables of the TensorFlow graph to a checkpoint after a certain number of epochs.
        if (epoch % CHECKPOINT_INTERVAL == 0) and args.save_checkpoint_path is not None:
            checkpoint.save(sess, save_path=save_path, global_step=epoch)
            print("Saved checkpoint for epoch {}".format(epoch))

    # Save weights
    if args.save_weight_path is not None:
        if not os.path.exists(args.save_weight_path):
            os.makedirs(args.save_weight_path)
        vgg_network.save_weights(args.save_weight_path, "sgd_weights", sess)
        print("Weights were saved in {}".format(args.save_weight_path + "sgd_weights.npz"))
