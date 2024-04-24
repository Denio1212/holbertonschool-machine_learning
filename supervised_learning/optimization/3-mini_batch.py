#!/usr/bin/env python3
"""
Trains a loaded neural network usning mini-batch gradient descent
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data
import tensorflow.compat.v1 as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network using mini-batch gradient descent

    :param X_train: numpy array of shape (m, 784) containing training data

    :param Y_train: one hot numpy array of shape (m, 10) containing
    training labels

    :param X_valid: numpy array of shape (m, 784) containing validation data

    :param Y_valid: one hot numpy array of shape (m, 10) containing
    validation labels

    :param batch_size: number of data points per batch

    :param epochs: number of epochs

    :param load_path: path to load model

    :param save_path: path to save model after training

    :return: The path where the model is saved
    """
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + '.meta')
        new_saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]

        for i in range(epochs + 1):

            train_cost, train_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_train,
                                                        y: Y_train})

            valid_cost, valid_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_valid,
                                                        y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if i < epochs:
                X_train_shuffled, Y_train_shuffled = shuffle_data(X_train,
                                                                  Y_train)

                num_batch = m // batch_size + (m % batch_size != 0)

                for step_num in range(num_batch):
                    start = step_num * batch_size
                    end = min(start + batch_size, m)

                    x_batch = X_train_shuffled[start:end]
                    y_batch = Y_train_shuffled[start:end]

                    sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

                    if step_num > 0 and (step_num + 1) % 100 == 0:
                        step_cost, step_acc = sess.run([loss, accuracy],
                                                       feed_dict={x: x_batch,
                                                                  y: y_batch})
                        print("\tStep {}:".format(step_num + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_acc))

            saved_model = new_saver.save(sess, save_path)

            return saved_model
