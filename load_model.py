
"""Contains functions for loading in a CNN with trained weights"""

import tensorflow as tf


def load_graph_pb(path):
    with tf.gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # TODO: change graphs so that backwards passes are possible no matter the size of the input etc.
        # for node in graph_def.node:
        #     if node.attr['padding'].s == "VALID":
        #         node.attr['padding'].s = str.encode("SAME")

        tf.import_graph_def(graph_def, name='')


def load_labels(path):

    return path
