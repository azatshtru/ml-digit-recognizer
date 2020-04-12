import struct
import numpy as np
from PIL import Image
import os

_datapath = "data/train-images-idx3-ubyte/train-images.idx3-ubyte"
_data_file = open(_datapath, 'rb')

def get_training_info(f):
    magic_number = f.read(4)
    print("magic number:", magic_number)

    num_images_bytes = f.read(4)
    num_images = struct.unpack('>i', num_images_bytes)
    print("No. of images:", num_images)

    num_rows = struct.unpack('>i', f.read(4))
    print("No. of rows:", num_rows)

    num_columns = struct.unpack('>i', f.read(4))
    print("No. of columns:", num_columns)

def get_training_data(f):

    os.mkdir("training_images")

    for j in range(1000):
        _image_array = np.zeros((28, 28), np.uint8)

        img_pxl_values = struct.unpack('>784B', f.read(784))

        for i in range(784):
            _image_array.itemset(i, img_pxl_values[i])

        img = Image.fromarray(_image_array)
        img.save("training_images/mnist {0}.png".format(j))

get_training_info(_data_file)
get_training_data(_data_file)
