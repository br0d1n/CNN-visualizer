import PIL.Image
import numpy as np

import transformations as trans
import visualize as vis


image_array = np.float32(PIL.Image.open("cat.jpg"))
padded_image = trans.pad(image_array, 50, constant_values=100)
rotated_image = trans.rotate(padded_image, 10)
rotated_image = trans.rotate_random(rotated_image)
scaled_image = trans.scale(rotated_image, 1.2)
saturated_image = trans.saturate_random(scaled_image)
jitter_image = trans.jitter(saturated_image, 50)
blurred_image = trans.blur(jitter_image, value=2)

vis.show_image(blurred_image)