
""" just a temporary entry-point for testing stuff. Nothing important here. """

import PIL.Image
import numpy as np
import tensorflow as tf

import transformations as trans
import visualize as vis
import load_model as load
import misc


model_path = "sample_models/inception/tensorflow_inception_graph.pb"
input_name = "input:0"

load.load_graph_pb(model_path)

vis.visualize_layer("mixed4a:0", input_name, steps=300)




