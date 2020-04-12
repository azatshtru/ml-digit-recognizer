#mnist_loader.py by azatshtru

import struct
import numpy as np
from PIL import Image
import os

class train_image_extractor(object):

    def __init__ (self):
        self._training_images = []
        self._training_labels = []

        self.validation_images = []
        self.validation_labels = []
    
    def get_training_info(self, f):
        magic_number = f.read(4)
        print("img magic number:", magic_number)

        num_images_bytes = f.read(4)
        num_images = struct.unpack('>i', num_images_bytes)
        print("No. of images:", num_images)

        num_rows = struct.unpack('>i', f.read(4))
        print("No. of rows:", num_rows)

        num_columns = struct.unpack('>i', f.read(4))
        print("No. of columns:", num_columns)

    def get_training_data_images(self, f, num_imgs, generate_images = False, gen_validate = False):

        if generate_images:
            os.mkdir("training_images")

        for j in range(num_imgs):
            img_pxl_values = struct.unpack('>784B', f.read(784))
            img_arr = np.zeros((784, 1), dtype = np.uint8)

            for m in range(784):
                img_arr.itemset(m, img_pxl_values[m])

            if(gen_validate == False):
                self._training_images.append(img_arr)
            else:
                self.validation_images.append(img_arr)

            if(generate_images):
                _image_array = np.zeros((28, 28), np.uint8)

                for i in range(784):
                    _image_array.itemset(i, img_pxl_values[i])

                img = Image.fromarray(_image_array)
                img.save("training_images/mnist {0}.png".format(j))

    def get_label_info(self, f):
        magic_number = f.read(4)
        print("label magic number:", magic_number)

        num_labels = struct.unpack('>i', f.read(4))
        print("No. of labels:", num_labels)

    def get_training_labels(self, f, num_lbls, gen_validate = False):
        for i in range(num_lbls):
            label = struct.unpack('>B', f.read(1))
            
            label_array = np.zeros((10, 1))
            label_array[label[0], 0] = 1

            if(gen_validate == False):                
                self._training_labels.append(label_array)
            else:                
                self.validation_labels.append(label_array)

            self.do_nothing(i)

    def do_nothing(self, x):
        return x