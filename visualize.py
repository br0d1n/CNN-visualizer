
"""This file contains functions for visualizing different parts of a CNN, in
addition to some other ones related to visualization"""

import tensorflow as tf
import PIL.Image


def show_image(image):
    image = tf.convert_to_tensor(image)
    with tf.Session():
        image = PIL.Image.fromarray(image.eval().astype('uint8'))
    image.show()

