#!/usr/bin/env python3
"""
Modified Lenet5 architecture
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """
    Implementation of LeNet-5 architecture using TensorFlow library.

    :param x: tf.placeholder shaped [m, 28, 28, 1] containing the input images

    :param y: tf.placeholder shaped [m, 10] containing the one hot labels
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv = tf.layers.conv2d(
        filters=6,
        kernel_size=5,
        padding='same',
        activation="relu"
    )(x)

    pool1 = tf.layers.max_pooling2d(pool_size=2,
                                    strides=2
                                    )(conv)

    conv2 = tf.layers.conv2d(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation="relu"
    )(pool1)

    pool2 = tf.layers.max_pooling2d(pool_size=2,
                                    strides=2
                                    )(conv2)

    flatten = tf.layers.flatten()(pool2)

    full1 = tf.layers.dense(
        units=120,
        activation="relu",
        kernel_initializer=initializer
    )(flatten)

    full2 = tf.layers.dense(
        units=84,
        activation="relu",
        kernel_initializer=initializer
    )(full1)

    full_out = tf.layers.dense(
        units=10,
        activation=None,
        kernel_initializer=initializer
    )(full2)

    softmax = tf.nn.softmax(full_out)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=full_out)

    train_step = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.argmax(full_out, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct = tf.equal(y_pred, y_true)

    correct_prediction = tf.cast(correct, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    return softmax, train_step, loss, accuracy
