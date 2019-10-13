"""
Utility functions.
"""

import numpy as np
import pickle


def unpickle(file_name):
    """
    Unpickles a pickled Python-object.
    """

    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def shuffle_data(X, y, one_hot_labels=True):
    """
    Shuffles the data in X, y with the same random permutation.
    Use one_hot_labels to specify whether the labels in y are
    in one-hot or index format.
    """

    n_examples = X.shape[0]

    perm = np.random.permutation(n_examples)
    X = X[perm, :, :]
    if one_hot_labels:
        y = y[perm, :]
    else:
        y = y[perm]

    return X, y


def split_data(X, y, train_frac, valid_frac, test_frac, shuffle=True):
    """
    Splits data into train/valid/test-sets according to the specified fractions.
    If shuffle is True, data is shuffled before splitting.
    """

    np.random.seed(1)

    assert train_frac + valid_frac + test_frac == 1, "Train/valid/test data fractions do not sum to one"

    n_examples = X.shape[0]

    # Shuffle
    if shuffle:
        X, y = shuffle_data(X, y)

    # Split
    ind_1 = int(np.round(train_frac * n_examples))
    ind_2 = int(np.round(ind_1 + valid_frac * n_examples))

    X_train = X[0:ind_1, :, :]
    y_train = y[0:ind_1, :]
    X_valid = X[ind_1:ind_2, :, :]
    y_valid = y[ind_1:ind_2, :]
    X_test = X[ind_2:, :, :]
    y_test = y[ind_2:, :]

    assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == n_examples, "Data split failed"

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)
