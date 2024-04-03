#!/usr/bin/env python3
"""
The Neural Network Deepens
"""

import numpy as np


class DeepNeuralNetwork:
    """
    The Deep Neural Network
    """

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):
            if not isinstance(layer, int) or layer < 1:
                raise TypeError('layers must be a list of positive integers')

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (np.random.randn(layer, previous) *
                                            np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: array with shape (nx, m) with input data
        nx is the number of input features
        m is the number of examples
        """
        self.__cache["A0"] = X

        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]

            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            a = 1 / (1 + np.exp(-z))

            self.__cache["A{}".format(index + 1)] = a
        return a, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: array with shape (1, m) with correct labels for input data
        :param A: array with shape (1, m) with activated  outputs
        for each example
        To avoid division by zero errors, we will use
        1.0000001 - A instead of 1 - A
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A) + (1 - Y) * np.log((1.0000001 - A))))
        costs = (1 / m) * (-m_loss)
        return costs

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates the Gradient Descent of one pass

        :param Y: array with shape (1, m) with correct labels for input data
        :param cache: dictionary with intermediary values of the network
        :param alpha: learning rate

        updates the private attributes __weights
        """
        m = Y.shape[1]
        back = {}

        for index in range(self.L, 0, -1):

            A = cache["A{}".format(index - 1)]
            if index == self.L:
                back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            else:
                dz_prev = back["dz{}".format(index + 1)]
                A_current = cache["A{}".format(index)]
                back["dz{}".format(index)] = (
                        np.matmul(W_prev.transpose(), dz_prev) *
                        (A_current * (1 - A_current)))

            dz = back["dz{}".format(index)]
            dW = (1 / m) * (np.matmul(dz, A.transpose()))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            W_prev = self.weights["W{}".format(index)]

            self.__weights["W{}".format(index)] = (
                    self.weights["W{}".format(index)] - (alpha * dW))
            self.__weights["b{}".format(index)] = (
                    self.weights["b{}".format(index)] - (alpha * db))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron and updates weights and cache

        :param X: array with input data, shape (nx, m)
        nx is the number of input samples
        m is the number of examples

        :param Y: array with shape (1, m) with the correct labels

        iterations: number of iterations
        if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer

        :param alpha: learning rate
        if alpha is not a float, raise a TypeError with the exception alpha must be a float
        if alpha is not positive, raise a ValueError with the exception alpha must be positive

        :param verbose: whether to or not print the training progress

        :param graph: whether to print the graph

        :param step: step size for gradient descent
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
