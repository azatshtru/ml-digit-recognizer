# neural network trainer using mnist_loader.py with network.py by azatshtru

import mnist_loader
import network

#getting training data
_datapath = "data/train-images-idx3-ubyte/train-images.idx3-ubyte"
_labelpath = "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte"

_data_file = open(_datapath, 'rb')
_label_file = open(_labelpath, 'rb')

mnist = mnist_loader.train_image_extractor()

mnist.get_training_info(_data_file)
mnist.get_training_data_images(_data_file, 50000)

mnist.get_label_info(_label_file)
mnist.get_training_labels(_label_file, 50000)

training_datapack = [(x, y) for (x, y) in zip(mnist._training_images, mnist._training_labels)]

mnist.get_training_data_images(_data_file, 10000, False, True)
mnist.get_training_labels(_label_file, 10000, True)

validation_datapack = [(x, y) for (x, y) in zip(mnist.validation_images, mnist.validation_labels)]

#using the data to train
nn = network.neural_network([784, 30, 10])

nn.initialize_sgd(training_datapack, 30, 10, 0.07, validation_data=validation_datapack)