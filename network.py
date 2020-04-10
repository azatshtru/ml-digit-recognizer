#network.py by azatshtru

import random
import numpy as np

class neural_network(object):

    def __init__ (self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[0:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for (y) in sizes[1:]]

    def feed_forward(self, x):
        a = x
        for (w, b) in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a