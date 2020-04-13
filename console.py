# console.py uses the correct weights and biases generated by network.py to classify images from test_images. 
# Authored by azatshtru.
import network
import indexer

import pickle
import struct
import os

import numpy as np
from PIL import Image

# getting test data.
with open('parameters/minima_weights.pkl', 'rb') as mnwt:
    data_weights = pickle.load(mnwt)

with open('parameters/minima_biases.pkl', 'rb') as mnbs:
    data_biases = pickle.load(mnbs)

# classifying images.
nn = network.neural_network([784, 30, 10])
nn.weights = data_weights
nn.biases = data_biases

def get_test_inputs ():
    indexer.get_test_images(100)

    input_image = int(input("Input the index of the image: "))
    test_sample = indexer.test_images[input_image]
    return test_sample

#test_sample_input = get_test_inputs()
test_sample_input = indexer.get_pixel_values("test_data_image.png")

result_list = nn.feed_forward(test_sample_input)

print(np.argmax(result_list))