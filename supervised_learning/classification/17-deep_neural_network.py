#!/usr/bin/env python3
""" Module creating a class NeuronalNetwork"""
import numpy as np


class DeepNeuralNetwork():
    """ Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initiates  Deep Neural Network class

        Inputs:
        nx - number of input features
            * must be integer of value greater than or equal to 1
        layers - number of nodes found in the each layer of the network
            * must be a list of positive integers

        Public Instance Attributes:
        L - The number of layers in the neural network
        cache - A dictionary holding all intermediary values of the network.
            Empty on instantiation
        weights - Dictionary holding all weights and biases of the network
        """

        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        layers_is_list_ints = isinstance(layers, list)

        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        if not layers_is_list_ints:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        # Private Properties
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        previous = nx
        weights_dict = {}

        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] < 0:
                raise TypeError("layers must be a list of positive integers")

            weights_dict["W{}".format(l + 1)] = (np.random.randn(layers[l],
                                                                 previous) *
                                                 np.sqrt(2 / previous))
            weights_dict["b{}".format(l + 1)] = np.zeros((layers[l], 1))
            previous = layers[l]

        self.__weights = weights_dict

    @property
    def L(self):
        """Getter for the private L attribute"""
        return self.__L

    @property
    def weights(self):
        """Getter for the private weights attribute"""
        return self.__weights

    @property
    def cache(self):
        """Getter for the private cache attribute"""
        return self.__cache