
"""Miscellaneous functions and random stuff goes here"""

import numpy as np
import random
import PIL.Image


# display image-tensor
def show_image(image):
    image = image.eval()
    image = np.clip(image / 255.0, 0, 1) * 255
    image = PIL.Image.fromarray(image.astype('uint8'))
    image.show()


# save image-tensor
def save_image(image, path):
    image = image.eval()
    image = np.clip(image / 255.0, 0, 1) * 255
    image = PIL.Image.fromarray(image.astype('uint8'))
    image.save(path)


# generate an image-array with random noise (from a gaussian distribution)
def random_noise_img(x_dim=200, y_dim=200):
    array = np.zeros((x_dim, y_dim, 3))
    for x in range(x_dim):
        for y in range(y_dim):
            array[x][y][0] = float(min(255, max(0, random.gauss(117, 50))))
            array[x][y][1] = float(min(255, max(0, random.gauss(117, 50))))
            array[x][y][2] = float(min(255, max(0, random.gauss(117, 50))))
    return array
