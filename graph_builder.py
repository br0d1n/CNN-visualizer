
""" Contains functions for loading in a CNN with trained weights, and
building the entire visualization graph """

import tensorflow as tf

import transformations as trans
import input_parameterization as par


def build(path, input_name, x_dim=200, y_dim=200, pad=None, jitter=None, rotate=None, scale=None):

    # load the pre-trained graph we are going to visualize
    graph_def = load_pb_graph(path)

    # parametrize the input space for better results
    # TODO: give the option to use other parametrization-spaces than just fourier
    fft_tensor = par.fft_img(1, x_dim, y_dim)

    # build a transformation graph, following the input-graph
    trans_graph = add_transforms(fft_tensor, pad, jitter, rotate, scale)
    trans_graph = tf.identity(trans_graph, name='transformed')

    # change the range
    lo, hi = (-117, 255 - 117)
    input_graph = lo + trans_graph * (hi - lo)

    # connect the graphs
    tf.import_graph_def(graph_def, {input_name: input_graph}, name='')

    # tensorboard stuff ..uncomment to take a look at the graph
    # logdir = "tensorboard/"
    # writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    tf.summary.merge_all()


def load_pb_graph(path):
    with tf.gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def add_transforms(tensor, pad, jitter, rotate, scale):
    if pad is not None:
        tensor = trans.pad(tensor, pad)
    if jitter is not None:
        tensor = trans.random_jitter(tensor, jitter)
    if rotate is not None:
        tensor = trans.random_rotate(tensor, rotate)
    if scale is not None:
        tensor = trans.random_scale(tensor, scale)
    return trans.random_jitter(tensor, 2)


def load_labels(path):
    pass
