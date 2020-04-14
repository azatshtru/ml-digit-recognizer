from PIL import Image
import numpy as np

import struct
import os

test_images = []

# Input an alpha image.
def get_pixel_values (img):
    arr = np.zeros((784, 1))
    img_arr = np.zeros((28, 28))

    with Image.open(img, 'r') as f:
        pixel_values = list(f.getdata())

    for i in range(784):
        arr.itemset(i, pixel_values[i])
        img_arr.itemset(i, pixel_values[i])

    _image = Image.fromarray(img_arr)
    _image.show()

    return arr

def get_test_images (req_imgs, generate_images=False):
    _path = "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
    f = open(_path, 'rb')

    magic_number = f.read(4)
    print(magic_number)

    num_images = struct.unpack('>i', f.read(4))
    print(num_images)

    num_rows = struct.unpack('>i', f.read(4))
    print(num_rows)

    num_columns = struct.unpack('>i', f.read(4))
    print(num_columns)

    if(generate_images): os.mkdir("test_images")

    for i in range(req_imgs):        
        img_array = np.zeros((28, 28), dtype=np.uint8)
        arr = np.zeros((784, 1), dtype=np.uint8)

        a = struct.unpack('>784B', f.read(784))

        for j in range(784):
            if(generate_images): img_array.itemset(j, a[j])
            arr.itemset(j, a[j])

        if(generate_images):
            img = Image.fromarray(img_array)
            img.save("test_images/mnist {0}.png".format(i))
        
        test_images.append(arr)

#Authored by azatshtru