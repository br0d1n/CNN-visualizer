
""" just a temporary entry-point for testing stuff. Nothing important here. """

import numpy as np

import visualize as vis
import misc
import PIL.Image


model_path = "sample_models/inception/tensorflow_inception_graph.pb"
input_name = "input:0"

# parameters which can be used in the random transformation-graph
pad = 16  # 16
jitter = 8  # 8
angles = list(range(-5, 5))  # (-5, 5)
scales = np.arange(0.9, 1.1, 0.1, dtype='float32')  # (0.9, 1.1, 0.1)

# some  optimization-objectives
opt_1 = ("mixed4d_3x3_bottleneck_pre_relu:0", 1)
opt_2 = ("mixed4d_3x3_bottleneck_pre_relu:0", 139)
opt_3 = ("mixed3a:0", None)
opt_mult = [opt_1, opt_2]
mix_1 = ("mixed4d:0", 42, 2.0)
mix_2 = ("mixed4d_3x3_bottleneck_pre_relu:0", 1, 1.0)
mix_3 = ("mixed4d_3x3_bottleneck_pre_relu:0", 139, 1.3)
mix = [mix_1, mix_2, mix_3]

# image that can be used for "deep dream" ..this is still a bit slow atm
cat = np.float32(PIL.Image.open("cat.jpg"))

# sample run of the visualize-function
results = vis.visualize_features(model_path, input_name, opt_3, x_dim=300, y_dim=300, steps=200,
                                 pad=pad, jitter=jitter, rotate=angles, scale=scales, dream_img=None)

# displaying the results
for img in results:
    misc.show_image(img)






