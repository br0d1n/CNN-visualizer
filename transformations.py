
"""This file contains functions for transforming tensorflow tensors that
represents images. These are used in order to get better-looking results
when performing feature-visualizations."""

import tensorflow as tf
import numpy as np
import math
import random
import PIL.Image
import PIL.ImageFilter


def pad(image, pixels, constant_values=117):
    padding = tf.constant([[pixels, pixels, ], [pixels, pixels], [0, 0]])
    image = tf.pad(image, padding, constant_values=constant_values)
    return image


def rotate(image, angle):
    dimensions = image.get_shape().as_list()
    height, width = dimensions[0], dimensions[1]
    image = pad(image, 40)  # pad sufficiently to avoid black lines
    radian = angle * math.pi / 180
    image = tf.contrib.image.rotate(image, radian)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def rotate_random(image, min_degree=-2, max_degree=2):
    dimensions = image.get_shape().as_list()
    height, width = dimensions[0], dimensions[1]
    image = pad(image, 40)
    angle = random.randint(min_degree, max_degree)
    radian = angle * math.pi / 180
    image = tf.contrib.image.rotate(image, radian)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def scale(image, factor):
    dimensions = image.get_shape().as_list()
    height, width = dimensions[0], dimensions[1]
    scale_height, scale_width = int(height * factor), int(width * factor)
    scale_size = tf.constant([scale_height, scale_width], tf.int32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, scale_size)
    image = tf.squeeze(image)
    image = pad(image, 20)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def scale_random(image, low_factor=0.99, high_factor=1.01):
    factor = random.uniform(low_factor, high_factor)
    dimensions = image.get_shape().as_list()
    height, width = dimensions[0], dimensions[1]
    scale_height, scale_width = int(height * factor), int(width * factor)
    scale_size = tf.constant([scale_height, scale_width], tf.int32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, scale_size)
    image = tf.squeeze(image)
    image = pad(image, 20)
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image


def jitter(image, max_pixels):
    pixels_x = random.randrange(-max_pixels, max_pixels+1)
    pixels_y = random.randrange(-max_pixels, max_pixels+1)
    dimensions = image.get_shape().as_list()

    width = 0 if pixels_x < 0 else dimensions[1]
    image = tf.concat([image[:, width - pixels_x:], image[:, :width - pixels_x]], axis=1)

    height = 0 if pixels_y < 0 else dimensions[0]
    image = tf.concat([image[height - pixels_y:, :], image[:height - pixels_y, :]], axis=0)

    return image


def blur(image, value=2):
    with tf.Session():
        image = PIL.Image.fromarray(image.eval().astype('uint8'))
    image = image.filter(PIL.ImageFilter.GaussianBlur(radius=value))
    image = np.float32(image)
    image = tf.convert_to_tensor(image)
    return image


def saturate_random(image, low_factor=0.8, high_factor=1.2):
    image = tf.image.random_saturation(image, low_factor, high_factor)
    return image
