
"""This file contains functions for transforming numpy arrays or tensorflow tensors
in different ways, in order to get better-looking results when performing
feature-visualizations.An array can represent an image, or simply a gradient which
is about to be applied to an image during an optimization process."""

import tensorflow as tf
import numpy as np
import math
import random
import PIL.Image, PIL.ImageFilter


def pad(image, pixels, constant_values=0):
    image = tf.convert_to_tensor(image)
    padding = tf.constant([[pixels, pixels, ], [pixels, pixels], [0, 0]])
    image = tf.pad(image, padding, constant_values=constant_values)
    return image


def rotate(image, angle):
    image = tf.convert_to_tensor(image)
    radian = angle * math.pi / 180
    image = tf.contrib.image.rotate(image, radian)
    return image


def rotate_random(image, min_degree=1, max_degree=5):
    image = tf.convert_to_tensor(image)
    angle = random.randint(min_degree, max_degree)
    radian = angle * math.pi / 180
    image = tf.contrib.image.rotate(image, radian)
    return image


def scale(image, factor):
    image = tf.convert_to_tensor(image)
    dimensions = image.get_shape().as_list()
    height, width = dimensions[0], dimensions[1]
    scale_height, scale_width = int(height * factor), int(width * factor)
    scale_size = tf.constant([scale_height, scale_width], tf.int32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, scale_size)
    image = tf.squeeze(image)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def scale_random(image, low_factor=0.9, high_factor=1.1):
    image = tf.convert_to_tensor(image)
    factor = random.uniform(low_factor, high_factor)
    dimensions = image.get_shape().as_list()
    height, width = dimensions[0], dimensions[1]
    scale_height, scale_width = int(height * factor), int(width * factor)
    scale_size = tf.constant([scale_height, scale_width], tf.int32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, scale_size)
    image = tf.squeeze(image)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def jitter(image, max_pixels):
    image = tf.convert_to_tensor(image)
    pixels_x = random.randrange(-max_pixels, max_pixels+1)
    pixels_y = random.randrange(-max_pixels, max_pixels+1)
    dimensions = image.get_shape().as_list()

    width = 0 if pixels_x < 0 else dimensions[1]
    image = tf.concat([image[:, width - pixels_x:], image[:, :width - pixels_x]], axis=1)

    height = 0 if pixels_y < 0 else dimensions[0]
    image = tf.concat([image[height - pixels_y:, :], image[:height - pixels_y, :]], axis=0)

    return image


def blur(image, value=2):
    image = tf.convert_to_tensor(image)
    with tf.Session():
        image = PIL.Image.fromarray(image.eval().astype('uint8'))
    image = image.filter(PIL.ImageFilter.GaussianBlur(radius=value))
    image = np.float32(image)
    image = tf.convert_to_tensor(image)
    return image


def saturate_random(image, low_factor=0.5, high_factor=1.5):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_saturation(image, low_factor, high_factor)
    return image

