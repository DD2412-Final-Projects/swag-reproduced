import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

from vgg16 import vgg16
from imagenet_classes import class_names

if __name__ == '__main__':
    """
    Example usage of the network.
    """
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, './../../weights/vgg16_weights.npz', sess, verbose=True)

    img1 = imread('./../../data/laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]

    print("\n--- Predictions ---")
    for p in preds:
        print(class_names[p], prob[p])
