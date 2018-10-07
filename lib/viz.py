'''Visualization.

'''

import logging

import imageio
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
from PIL import Image

logger = logging.getLogger('BGAN.viz')

_options = dict(
    use_tanh=False,
    quantized=False,
    img=None
)


def setup(use_tanh=None, quantized=None, img=None):
    global _options
    if use_tanh is not None:
        _options['use_tanh'] = use_tanh
    if quantized is not None:
        _options['quantized'] = quantized
    if img is not None:
        _options['img'] = img


def dequantize(images):
    images = np.argmax(images, axis=1).astype('uint8')
    images_ = []
    for image in images:
        img2 = Image.fromarray(image)
        img2.putpalette(_options['img'].getpalette())
        img2 = img2.convert('RGB')
        images_.append(np.array(img2))
    images = np.array(images_).transpose(0, 3, 1, 2)
    return images


def save_images(images, num_x, num_y, out_file=None):
    import image_saver
    import output
    output.to_protobuf(images.transpose(0, 2, 3, 1), num_samples=1)
    image_saver.save_image("out/proto/", "out/images")


def save_movie(images, num_x, num_y, out_file=None):
    pass