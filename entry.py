
""" just a temporary entry-point for testing stuff. Nothing important here. """

import numpy as np

import visualize as vis
import misc

# lists of available parameters, which be used in the random transformation-graph
pad = 16  # 16
jitter = 8  # 8
angles = list(range(-5, 5))  # (-5, 5)
scales = np.arange(0.9, 1.1, 0.1, dtype='float32')  # (0.9, 1.1, 0.1)


model_path = "sample_models/inception/tensorflow_inception_graph.pb"
input_name = "input:0"

opt_1 = ("mixed4d_3x3_bottleneck_pre_relu:0", 1)

# vis.visualize_layer(model_path, input_name, opt_1, x_dim=300, y_dim=300, steps=200, pad=pad, jitter=jitter, rotate=angles, scale=scales)

opt_2 = ("mixed4d_3x3_bottleneck_pre_relu:0", 139)
opt_3 = [opt_1, opt_2]

images = vis.visualize_layer(model_path, input_name, opt_3, x_dim=150, y_dim=150, steps=100, pad=5, jitter=jitter, rotate=angles, scale=scales)
for img in images:
    misc.show_image(img)

# TODO: make the following examples work
# mix_1 = ("mixed4a:0", 476, 0.4)
# mix_2 = ("mixed2b:0", None, 0.6)
# mix = [mix_1, mix_2]
#
# vis.visualize_layer(model_path, input_name, mix, x_dim=200, y_dim=200, steps=100)
#
# img = some_image
#
# vis.deep_dream(model_path, input_name, mix, img, x_dim=200, y_dim=200, steps=100)



