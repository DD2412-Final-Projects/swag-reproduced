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
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

import utils
from networks.vgg16 import VGG16

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

BATCH_SIZE = 128
S = 30  # number of samples to take from the SWAG-distribution


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", metavar="PATH TO DATA", default=None,
                        help="Path to data that has been preprocessed using preprocess_data.py.")
    parser.add_argument("--load_param_file", dest="load_param_file", metavar="LOAD PARAMETER FILE", default=None,
                        help="File to load trained SWAG parameters for the network from.")
    parser.add_argument("--mode", dest="mode", metavar="MODE", default="swag",
                        help="Choose between 'swa', 'swag-diag', 'swag', or 'sgd-noise'. Defaults to swag.")
    parser.add_argument("--noise", dest="noise", metavar="NOISE", type=float, default=0.1,
                        help="Amount of noise in sgd-noise. Default: 0.1")

    args = parser.parse_args()
    assert args.data_path is not None, "Data path must be specified."
    assert args.load_param_file is not None, "SWAG parameter file must be specified."
    assert args.mode in ["swa", "swag-diag", "swag", "sgd-noise"], "SWAG type argument must be either 'swa', 'swag-diag', 'swag', or 'sgd-noise'"

    return args


def reliability_diagram(y_pred, y_true, n_sample, n=20):
    """
    Creates a reliability diagram with the method described in the SWAG paper.

    1. y_pred is sorted based on confidence and split into n bins.
    2. Mean confidence and mean accuracy is computed for each bin.
    3. (mean confidence - mean accuracy) is plotted against the maximum confidence for each bin.
    """

    # Sort
    y_pred_true = list(zip(list(y_pred), list(y_true)))
    y_pred_true.sort(key=lambda arr_tuple: np.amax(arr_tuple[0]))
    y_pred_sorted, y_true_sorted = list(zip(*y_pred_true))

    # Split into bins
    y_pred_binned = np.array_split(np.asarray(y_pred_sorted), n)  # list of arrays
    y_true_binned = np.array_split(np.asarray(y_true_sorted), n)  # list of arrays
    bin_weight = [_.shape[0] / n_sample for _ in y_true_binned]

    # Compute mean confidence, mean accuracy and max confidence for each bin
    mean_confidence_per_bin = [np.mean(np.amax(y_preds, axis=1)) for y_preds in y_pred_binned]
    mean_accuracy_per_bin = [np.mean(np.equal(np.argmax(y_p, axis=1), np.argmax(y_t, axis=1)))
                             for y_p, y_t in zip(y_pred_binned, y_true_binned)]
    max_confidence_per_bin = [np.amax(y_p) for y_p in y_pred_binned]
    mean_conf_acc_diff_per_bin = [conf - acc for conf, acc in zip(mean_confidence_per_bin, mean_accuracy_per_bin)]

    ece = sum([np.abs(a) * np.abs(b) for a, b in zip(bin_weight, mean_conf_acc_diff_per_bin)])

    # Plot results
    plt.figure()
    ax = plt.axes()
    ax.axhline(linestyle="--", color="b")
    ax.plot(max_confidence_per_bin, mean_conf_acc_diff_per_bin, "-o", color="r")
    plt.legend(labels=["Ideal", "Result"])
    plt.title("Reliability diagram")
    plt.xlabel("Confidence (max in bin)")
    plt.ylabel("Confidence - Accuracy (mean in bin)")
    plt.xscale('logit')
    plt.xlim(0.01, 0.9999999)
    ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.grid()
    # plt.show()

    return ece, max_confidence_per_bin, mean_conf_acc_diff_per_bin


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
    vgg_network = VGG16(X_input, n_classes, weights=None, sess=sess, dropout=0, augment_inputs=False)
    logits = vgg_network.fc3l  # Output of the final layer

    # Define evaluation metrics
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input))
    predictions = tf.nn.softmax(logits)
    predicted_correctly = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    y_pred = np.empty((0, n_classes))

    # Load the SWAG-weight parameters
    param_dict = np.load(args.load_param_file)

    # Sample from the distribution of weights, compute predictions
    # and keep a sum of predictions across sampled weights
    if args.mode == "swa":
        S = 1
    y_pred_sum = np.zeros((n_samples, n_classes))
    print("Computing test performance..")
    for s in tqdm(range(S)):

        # Sample weights
        if args.mode == "swag":
            d = param_dict["theta_SWA"].shape[0]
            K = param_dict["K_SWAG"]
            z1 = np.random.normal(np.zeros((d,)), np.ones((d,)))  # z1 ~ N(0, I_d)
            z2 = np.random.normal(np.zeros((K,)), np.ones((K,)))  # z2 ~ N(0, I_K)
            sigma_SWAG = np.clip(param_dict["sigma_SWAG"], a_min=1e-30, a_max=None)
            weight_sample = param_dict["theta_SWA"] + (1 / np.sqrt(2)) * np.multiply(np.sqrt(sigma_SWAG), z1) + \
                (1 / np.sqrt(2 * (K - 1))) * np.dot(param_dict["D_SWAG"], z2)
        elif args.mode == "swag-diag":
            d = param_dict["theta_SWA"].shape[0]
            z1 = np.random.normal(np.zeros((d,)), np.ones((d,)))  # z1 ~ N(0, I_d)
            sigma_SWAG = np.clip(param_dict["sigma_SWAG"], a_min=1e-30, a_max=None)
            weight_sample = param_dict["theta_SWA"] + np.multiply(np.sqrt(sigma_SWAG), z1)
        elif args.mode == "swa":
            weight_sample = param_dict["theta_SWA"]
        elif args.mode == "sgd-noise":
            weight_dict = {}
            for k in param_dict.keys():
                weight_dict[k] = param_dict[k] + np.random.normal(np.zeros(param_dict[k].shape), args.noise * np.absolute(param_dict[k]))

        # Load the weight sample into the network
        if args.mode != "sgd-noise":
            weight_dict = vgg_network.unflatten_weights(weight_sample)
        vgg_network.load_weights(weight_dict, sess)

        for step in range(int(np.ceil(n_samples / BATCH_SIZE))):

            X_batch = X_test[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            y_batch = y_test[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]

            preds = sess.run(predictions, feed_dict={X_input: X_batch, y_input: y_batch})
            y_pred_sum[step * BATCH_SIZE: (step + 1) * BATCH_SIZE, :] += preds

    # Compute final predictions
    y_pred = (1 / S) * y_pred_sum

    # Display results
    loss_test = log_loss(y_true=y_test, y_pred=y_pred)
    acc_test = accuracy_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y_pred, axis=1))
    print("\n---- Test Results ----")
    print('Loss: {} \nAccuracy: {}'.format(loss_test, acc_test))
    ece, max_conf, conf_acc_diff = reliability_diagram(y_pred, y_test, n_samples)
    print('ECE:', ece)
    print("\n To recreate reliability diagram:")
    print("max(conf): ", max_conf)
    print("conf - acc: ", conf_acc_diff)
