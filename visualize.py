
""" This file contains functions for visualizing different parts of a CNN, in
addition to some other ones related to visualization """

import tensorflow as tf
import time

import graph_builder
import misc


def visualize_features(model_path, input_name, opt, x_dim=128, y_dim=128, steps=150, lr=0.06,
                    pad=None, jitter=None, rotate=None, scale=None, optimizer=None, dream_img=None):

    num_visualizations = 1
    mix = False
    if isinstance(opt, list):
        if len(opt[0]) == 2:
            num_visualizations = len(opt)
        elif len(opt[0]) == 3:
            mix = True

    # start the session
    with tf.Graph().as_default() as graph, tf.Session() as sess:

        # build the entire graph (parametrisation of the input space + transforms + the imported graph)
        graph_builder.build(model_path, input_name, x_dim, y_dim, pad, jitter, rotate, scale, dream_img)

        # select tensors that we might want to access later
        image_tensor = graph.get_tensor_by_name('image:0')
        trans_tensor = graph.get_tensor_by_name('transformed:0')

        # create the optimizer to the graph
        if dream_img is not None: lr = 3
        optimizer = optimizer or tf.train.AdamOptimizer(learning_rate=lr)

        # train the network to optimize the image(s)
        start_time = time.time()
        images = []
        for n in range(num_visualizations):

            # create the loss function
            if num_visualizations > 1:
                loss = create_loss(opt[n], graph)
            elif mix:
                loss = create_loss(opt, graph)
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
                if dream_img is None:
                    misc.save_image(img, "out/test" + str(i) + ".jpg")
                else:
                    misc.save_image_naive(img, "out/test" + str(i) + ".jpg")

                # optimize the image a little bit
                sess.run([loss, opt_tensor])

            img = image_tensor.eval()[0]
            images.append(img)

        duration = time.time() - start_time
        print("visualization complete\ttime:", duration)
        return images


def create_loss(opt, graph):

    if isinstance(opt, tuple):
        layer_name = opt[0]
        channel = opt[1]

        layer_tensor = graph.get_tensor_by_name(layer_name)
        loss = tf.reduce_mean(layer_tensor[:, :, :, channel])

    elif isinstance(opt, list):
        layer_tensor = graph.get_tensor_by_name(opt[0][0])
        channel_tensor = layer_tensor[:, :, :, opt[0][1]]
        loss = tf.reduce_mean(channel_tensor * opt[0][2])
        for i in range(1, len(opt)):
            layer_tensor = graph.get_tensor_by_name(opt[i][0])
            channel_tensor = layer_tensor[:, :, :, opt[i][1]]
            loss_inner = tf.reduce_mean(channel_tensor * opt[i][2])
            loss += loss_inner

    return loss



