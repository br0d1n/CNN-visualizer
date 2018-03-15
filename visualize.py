
"""This file contains functions for visualizing different parts of a CNN, in
addition to some other ones related to visualization"""

import tensorflow as tf
import numpy as np
import random

import transformations as trans
import misc


def compute_gradient(image, gradient_function, input_name, sess):

    image = tf.expand_dims(image, axis=0)
    feed_dict = {input_name: image.eval()}
    gradient = sess.run(gradient_function, feed_dict)[0]
    gradient = gradient / (1e-8 + np.std(gradient))

    return gradient


# Need to use this thing if images are too large for memory
def compute_gradient_from_image_segments(image, gradient_function, input_name, sess, segment_dim=200):

    # array containing zeroes, which will be filled up with gradients from each segment
    gradient = np.zeros_like(image)
    y_max = len(gradient)
    x_max = len(gradient[0])

    # the starting point of the grid we use to segment the image
    # to divide the picture differently each time, we introduce some randomness
    y_start = random.randint(-segment_dim+1, 0)
    x_origin = random.randint(-segment_dim+1, 0)
    while y_start < y_max:

        # find the beginning and end of the current segment. Can't be outside the image
        y_end = min(y_start + segment_dim, y_max)
        y_start = max(0, y_start)

        # randomness in the x-direction
        x_start = x_origin
        while x_start < x_max:

            x_end = min(x_start + segment_dim, x_max)
            x_start = max(0, x_start)

            # get the current image-segment, witch we will compute the gradient for
            image_segment = image[y_start:y_end, x_start:x_end, :]

            # compute the gradient for the current segment
            segment_gradient = compute_gradient(image_segment, gradient_function, input_name, sess)

            # adding the segment-gradient to the gradient for the entire image
            gradient[y_start:y_end, x_start:x_end, :] = segment_gradient
            x_start = x_end

        # continue the next segment, where the last one ended
        y_start = y_end

    return gradient


def visualize_layer(layer_name, input_name, steps=200, step_size=3):

    with tf.Session() as sess:
        layer_tensor = sess.graph.get_tensor_by_name(layer_name)
        layer_tensor = tf.square(layer_tensor)
        input_tensor = sess.graph.get_tensor_by_name(input_name)
        layer_mean = tf.reduce_mean(layer_tensor[:, :, :, :])
        gradient_function = tf.gradients(layer_mean, input_tensor)[0]

        optimized_image = misc.random_noise_img(200)
        optimized_image = tf.convert_to_tensor(optimized_image, dtype=np.float32)
        optimized_image = trans.pad(optimized_image, 16)

        for i in range(steps):
            print (i)

            optimized_image = trans.jitter(optimized_image, 8)
            optimized_image = trans.scale_random(optimized_image)
            optimized_image = trans.rotate_random(optimized_image)
            optimized_image = trans.jitter(optimized_image, 2)

            gradient = compute_gradient(optimized_image, gradient_function, input_name, sess)

            optimized_image = tf.add(optimized_image, tf.multiply(gradient, step_size))

            if i % 2 == 0:

                misc.save_image(optimized_image, "out/test"+str(i)+".jpg")

        misc.show_image(optimized_image)

