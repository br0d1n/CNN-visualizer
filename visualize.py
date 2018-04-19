
""" This file contains functions for visualizing different parts of a CNN, in
addition to some other ones related to visualization """

import tensorflow as tf
import time

import graph_builder
import misc


# TODO: fix this
def visualize_layer(model_path, input_name, opt, x_dim=128, y_dim=128, steps=150,
                    pad=None, jitter=None, rotate=None, scale=None, optimizer=None):

    num_visualizations = 1
    if isinstance(opt, list) and len(opt[0]) == 2:
        num_visualizations = len(opt)

    # start the session
    with tf.Graph().as_default() as graph, tf.Session() as sess:

        # build the entire graph (parametrisation of the input space + transforms + the imported graph)
        graph_builder.build(model_path, input_name, x_dim, y_dim, pad, jitter, rotate, scale)

        # select tensors that we might want to access later
        image_tensor = graph.get_tensor_by_name('image:0')
        trans_tensor = graph.get_tensor_by_name('transformed:0')

        # create the optimizer to the graph
        optimizer = optimizer or tf.train.AdamOptimizer(learning_rate=0.06)

        # train the network to optimize the image(s)
        start_time = time.time()
        images = []
        for n in range(num_visualizations):

            # create the loss function
            # TODO: make it easy to mix different layers etc. here
            if num_visualizations > 1:
                loss = create_loss(opt[n], graph)
            else:
                loss = create_loss(opt, graph)

            # add the optimizer
            opt_tensor = optimizer.minimize(-loss)

            # initalize variables
            sess.run(tf.global_variables_initializer())

            for i in range(steps):
                print(i)

                # save the current optimized image (for testing and cool animations)
                img = image_tensor.eval()[0]
                misc.save_image(img, "out/test" + str(i) + ".jpg")

                # optimize the image a little bit
                sess.run([loss, opt_tensor])

            img = image_tensor.eval()[0]
            images.append(img)

        duration = time.time() - start_time
        print("visualization complete\ttime:", duration)
        return images


def create_loss(opt, graph):

    layer_name = None
    channel = None

    # find the type of objective we are trying to optimize for
    if isinstance(opt, str):
        layer_name = opt
    elif isinstance(opt, tuple):
        layer_name = opt[0]
        channel = opt[1]

    layer_tensor = graph.get_tensor_by_name(layer_name)
    loss = tf.reduce_mean(layer_tensor[:, :, :, channel])

    return loss



