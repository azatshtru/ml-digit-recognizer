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

    def initialize_sgd (self, training_data, epochs, mini_batch_size, eta):
        #training_data is a list of tuples (x, y) where x is the vertical vector/array of input pixel values b/w 0 and 1 and y is
        #desired output array/vector.

        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: #mini_batch is an element of mini_batches which are sliced sub-lists of training_data
                # hence mini_batch is a list of tuples (x, y) with x the input vector, y the desired output vector
                self.calculate_descent(mini_batch, eta)
            print(i)
    
    def calculate_descent(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for(x, y) in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [(nw + dnw) for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
            nabla_b = [(nb + dnb) for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
        
        self.weights = [(w - ((eta / len(mini_batch)) * nw)) for (w, nw) in zip(self.weights, nabla_w)]
        self.biases = [(b - ((eta / len(mini_batch)) * nb)) for (b, nb) in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        d_nabla_w = [np.zeros(w.shape) for w in self.weights]
        d_nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []

        #feeding_forward to calculate activations and zs of all hidden layers and output layer w.r.t current weights and biases 
        for (w, b) in zip(self.weights, self.biases):
            z = (np.dot(w, activation) + b)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #error for the last layer
        _error = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        d_nabla_w[-1] = np.dot(_error, activations[-2].transpose())
        d_nabla_b[-1] = _error

        #errors for previous layers
        for j in range(2, self.num_layers):
            _error = np.dot(self.weights[-j + 1].transpose(), _error) * sigmoid_prime(zs[-j])
            d_nabla_w[-j] = np.dot(_error, activations[-j - 1].transpose())
            d_nabla_b[-j] = _error

        return d_nabla_w, d_nabla_b

def cost_derivative(a, y):
    diff = a - y
    return diff

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def sigmoid_prime(z):
    a = sigmoid(z) * (1 - sigmoid(z))
    return a