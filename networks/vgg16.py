########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np


class VGG16:

    def __init__(self, imgs, n_classes, weights=None, sess=None):
        """
        Initializes the VGG16 network.
        """

        self.imgs = imgs
        self.n_classes = n_classes
        self.conv_initializer = tf.contrib.layers.variance_scaling_initializer()  # He initilization
        self.fc_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.weight_keys = []
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights_from_file(weights, sess)

        conv_out_shape = int(np.prod(self.pool5.get_shape()[1:]))

        self.VGG16_VAR_DIMS = {
            "conv1_1_W": (3, 3, 3, 64), "conv1_1_b": (64),
            "conv1_2_W": (3, 3, 64, 64), "conv1_2_b": (64),
            "conv2_1_W": (3, 3, 64, 128), "conv2_1_b": (128),
            "conv2_2_W": (3, 3, 128, 128), "conv2_2_b": (128),
            "conv3_1_W": (3, 3, 128, 256), "conv3_1_b": (256),
            "conv3_2_W": (3, 3, 256, 256), "conv3_2_b": (256),
            "conv3_3_W": (3, 3, 256, 256), "conv3_3_b": (256),
            "conv4_1_W": (3, 3, 256, 512), "conv4_1_b": (512),
            "conv4_2_W": (3, 3, 256, 512), "conv4_2_b": (512),
            "conv4_3_W": (3, 3, 512, 512), "conv4_3_b": (512),
            "conv5_1_W": (3, 3, 512, 512), "conv5_1_b": (512),
            "conv5_2_W": (3, 3, 512, 512), "conv5_2_b": (512),
            "conv5_3_W": (3, 3, 512, 512), "conv5_3_b": (512),
            "fc1_W": (conv_out_shape, 512), "fc1_b": (512),
            "fc2_W": (512, 512), "fc2_b": (512),
            "fc3_W": (512, self.n_classes), "fc3_b": (self.n_classes)
        }

    def convlayers(self):
        """
        Adds the convolution layers to the network.
        """
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            images = self.imgs / 255
            mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            stddev = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32, shape=[1, 1, 1, 3], name="img_stddev")
            images = tf.map_fn(lambda image: self.distort_image(image), images)
            images = (images - mean) / stddev

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 3, 64]), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv1_1_W", "conv1_1_b"]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 64, 64]), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv1_2_W", "conv1_2_b"]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 64, 128]), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv2_1_W", "conv2_1_b"]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 128, 128]), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv2_2_W", "conv2_2_b"]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 128, 256]), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv3_1_W", "conv3_1_b"]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 256, 256]), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv3_2_W", "conv3_2_b"]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 256, 256]), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv3_3_W", "conv3_3_b"]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 256, 512]), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv4_1_W", "conv4_1_b"]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 512, 512]), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv4_2_W", "conv4_2_b"]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 512, 512]), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv4_3_W", "conv4_3_b"]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 512, 512]), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv5_1_W", "conv5_1_b"]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 512, 512]), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv5_2_W", "conv5_2_b"]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(self.conv_initializer([3, 3, 512, 512]), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv5_3_W", "conv5_3_b"]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def fc_layers(self):
        """
        Adds the fully connected layers to the network.
        """
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(self.fc_initializer([shape, 512]), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
            self.weight_keys += ["fc1_W", "fc1_b"]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(self.fc_initializer([512, 512]), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]
            self.weight_keys += ["fc2_W", "fc2_b"]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(self.fc_initializer([512, self.n_classes]), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.n_classes], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]
            self.weight_keys += ["fc3_W", "fc3_b"]

    def load_weights_from_file(self, weight_file, sess):
        """
        Initializes the weights of the network to the values in weight_file.
        """
        weight_dict = np.load(weight_file)
        self.load_weights(weight_dict, sess)

    def load_weights(self, weight_dict, sess):
        """
        Initializes the weights of the network to the values in weight_dict.
        (weight_dict needs to be a dictionary structured as what is returned by
        self.get_weights).
        """
        keys = sorted(weight_dict.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weight_dict[k]))

    def save_weights(self, weight_path, weight_file_name, sess):
        """
        Saves the current weights of the network to a file with the name
        weight_file_name in weight_path.
        """
        weight_dict = self.get_weights(sess)
        np.savez(weight_path + weight_file_name, **weight_dict)

    def get_weights(self, sess):
        """
        Returns the current values of all weights in the network
        in a dictionary with the same keys as self.weight_keys.
        """
        keys = sorted(self.weight_keys)
        weight_dict = {}
        for i, k in enumerate(keys):
            weight_dict[k] = sess.run(self.parameters[i])

        return weight_dict

    def get_weights_flat(self, sess):
        """
        Returns the current values of all weights in the network
        in a single flattened vector.

        Weights are ordered alphabetically after their key (e.g. conv1_1_W) and
        flattened in row major order.
        """
        weight_dict = self.get_weights(sess)
        keys = sorted(weight_dict.keys())
        weight_vector = []

        for key in keys:
            weight_vector.append(weight_dict[key].flatten())

        return np.concatenate(weight_vector)

    def unflatten_weights(self, weight_vector):
        """
        Takes a vector of weights (structured as the return value of self.get_weights_flat)
        and "unflattens" them into a weight_dict (like the one returned by get_weights).
        """
        keys = sorted(self.VGG16_VAR_DIMS.keys())
        weight_dict = {}
        slice_index = 0

        for key in keys:
            dims = self.VGG16_VAR_DIMS[key]
            size = np.prod(dims)
            values = weight_vector[slice_index: slice_index + size]
            slice_index += size

            weight_dict[key] = values.reshape(dims)

        return weight_dict

    @staticmethod
    def distort_image(image):

        """
        THe followings are done:
        - zero-padded with 4 pixels on each side
        - randomly crop the images to same size but distortet
        - Randomly flip the 50% image horizontally.
        """
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_image_with_crop_or_pad(image, 32 + 4, 32 + 4)
        image = tf.random_crop(image, size=[32, 32, 3])
        return image
