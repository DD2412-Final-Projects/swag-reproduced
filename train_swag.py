"""
Trains a model using SWA and saves the averaged weight parameters.

1. Load/import the network architecture from the 'networks' directory.
2. Load training data.
3. Train the model with SWA.
4. Save the learned weight parameters.
"""
import argparse
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

import utils
from networks.vgg16 import VGG16

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Hyperparameters
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 300
BATCH_SIZE = 128
DISPLAY_INTERVAL = 10  # How often to display loss/accuracy during training (steps)
CHECKPOINT_INTERVAL = 50  # How often to save checkpoints (epochs)
SWAG_START_EPOCH = 160  # after how many epochs of training to start collecting samples for SWAG
K_SWAG = 20  # maximum number of columns in deviation matrix D


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data that has been preprocessed using preprocess_data.py.")
    parser.add_argument("--save_param_path", dest="save_param_path", metavar="SAVE PARAMETER PATH", default=None,
                        help="Path to save trained SWAG parameters for the network to.")
    parser.add_argument("--save_checkpoint_path", dest="save_checkpoint_path", metavar="SAVE CHECKPOINT PATH", default=None,
                        help="Path to save checkpoints to")
    parser.add_argument("--load_checkpoint_path", dest="load_checkpoint_path", metavar='LOAD CHECKPOINT PATH', default=None,
                        help="Path to load checkpoint from.")
    parser.add_argument("--save_plots_path", dest="save_plots_path", metavar="SAVE PLOTS PATH", default=None,
                        help="Path to save plots to.")

    args = parser.parse_args()
    assert args.data_path is not None, "Data path must be specified."

    return args


def plot_cost(c_v, c_t, save_plots_path):
    """
    Creates a plot of validation cost c_v
    and displays it.
    """

    plt.figure()
    plt.plot(c_v, label='Validation loss')
    plt.plot(c_t, label='Training loss')
    plt.legend()
    title = 'Loss per epoch'
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_plots_path + "swag_loss_plot.png")


def plot_acc(acc_v, acc_t, save_plots_path):
    """
    Creates a plot of validation cost c_v
    and displays it.
    """

    plt.figure()
    plt.plot(acc_v, label='Validation acc')
    plt.plot(acc_t, label='Training acc')
    plt.legend()
    title = 'Accuracy per epoch'
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(save_plots_path + "swag_accuracy_plot.png")


def save_swag_params(save_path, param_dict):
    """
    Saves the SWAG parameters in param_dict to save_path
    in a compressed .npz file with the name swag_params.npz
    """
    np.savez(save_path + "swag_params.npz", **param_dict)


if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    # Load training data and validation data
    X_train = np.load(args.data_path + "X_train.npy")
    X_val = np.load(args.data_path + "X_valid.npy")

    y_train = np.load(args.data_path + "y_train.npy")
    y_val = np.load(args.data_path + "y_valid.npy")
    n_samples, n_classes = y_train.shape
    width, height, n_channels = X_train.shape[1:]

    # Load network architecture
    sess = tf.Session()
    X_input = tf.placeholder(tf.float32, [None, width, height, n_channels])
    y_input = tf.placeholder(tf.float32, [None, n_classes])
    vgg_network = VGG16(X_input, n_classes, weights=None, sess=sess)
    logits = vgg_network.fc3l  # Output of the final layer

    # Define loss and optimizer
    weights = tf.trainable_variables()
    l2_reg = WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(w) for w in weights])
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input))
    loss = ce_loss + l2_reg
    optimizer = tf.contrib.optimizer_v2.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
    train_operation = optimizer.minimize(loss)

    # Define evaluation metrics
    predictions = tf.nn.softmax(logits)
    predicted_correctly = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))

    # Create checkpoint object if loading/saving checkpoints
    if args.save_checkpoint_path is not None or args.load_checkpoint_path is not None:
        checkpoint = tf.train.Saver()

    # Create checkpoint path is saving checkpoints
    if args.save_checkpoint_path is not None:
        save_dir = args.save_checkpoint_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'model')

    # Load checkpoint
    if args.load_checkpoint_path is not None:
        print("Trying to restore last checkpoint ...")

        # Use TensorFlow to find the latest checkpoint - if any.
        last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=args.load_checkpoint_path)

        # Try and load the data in the checkpoint.
        checkpoint.restore(sess, save_path=last_ckpt_path)

        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_ckpt_path)

    # Initialize weights if not loading checkpoint
    else:
        sess.run(tf.initialize_all_variables())

    # initialization for plots
    validation_loss, validation_acc = [], []
    training_loss, training_acc = [], []

    # Initialization of SWAG-parameters
    first_moment = vgg_network.get_weights_flat(sess)
    second_moment = first_moment ** 2
    D = np.empty((first_moment.shape[0], 0))  # deviation matrix
    global_step = 0  # total number of iterations (across different epochs)

    # Run training with SWAG
    n_SWAG = 1  # number of SWAG samples
    for epoch in range(EPOCHS):

        print("\n---- Epoch {} ----\n".format(epoch + 1))
        for step in range(n_samples // BATCH_SIZE):

            X_batch = X_train[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            y_batch = y_train[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]

            sess.run(train_operation, feed_dict={X_input: X_batch, y_input: y_batch})

            if step % DISPLAY_INTERVAL == 0:
                loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X_input: X_batch, y_input: y_batch})
                print("Iteration {}, Batch loss = {}, Batch accuracy = {}".format(step + 1, loss_val, acc_val))

        # Perform SWAG-update
        if epoch >= SWAG_START_EPOCH:
            new_weights = vgg_network.get_weights_flat(sess)

            first_moment = (n_SWAG * first_moment + new_weights) / (n_SWAG + 1)
            second_moment = (n_SWAG * second_moment + new_weights ** 2) / (n_SWAG + 1)

            if D.shape[1] == K_SWAG:
                D = np.delete(D, 0, 1)  # remove first column
            new_D_col = second_moment - first_moment ** 2
            D = np.append(D, new_D_col.reshape(new_D_col.shape[0], 1), axis=1)

            n_SWAG += 1

        # Storing validation data for plotting
        X_train, y_train = utils.shuffle_data(X_train, y_train)
        X_valid, y_valid = utils.shuffle_data(X_val, y_val)
        v_loss, v_acc = sess.run([loss, accuracy], feed_dict={X_input: X_val[:1000], y_input: y_val[:1000]})
        tr_loss, tr_acc = sess.run([loss, accuracy], feed_dict={X_input: X_batch, y_input: y_batch})
        validation_loss.append(v_loss)
        validation_acc.append(v_acc)
        training_loss.append(tr_loss)
        training_acc.append(tr_acc)

        # Save all variables of the TensorFlow graph to a checkpoint after a certain number of epochs.
        if ((epoch + 1) % CHECKPOINT_INTERVAL == 0) and args.save_checkpoint_path is not None:
            checkpoint.save(sess, save_path=save_path, global_step=epoch)
            print("Saved checkpoint for epoch {}".format(epoch))

    # Compute SWAG parameters
    param_dict = {}
    param_dict["theta_SWA"] = first_moment
    param_dict["sigma_SWAG"] = second_moment - first_moment ** 2  # NOTE: stored as vector for efficiency
    param_dict["D_SWAG"] = D
    param_dict["K_SWAG"] = K_SWAG

    # Save SWAG-parameters
    save_swag_params(args.save_param_path, param_dict)

    # Plot validation stats
    if args.save_plots_path is not None:
        plot_cost(validation_loss, training_loss, args.save_plots_path)
        plot_acc(validation_acc, training_acc, args.save_plots_path)
    else:
        plot_cost(validation_loss, training_loss)
        plot_acc(validation_acc, training_acc)
