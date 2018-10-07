'''DCGAN for 28x28 images. Used for MNIST. Published version.

'''

import logging

from lasagne.layers import (
    batch_norm, Conv2DLayer, DenseLayer, InputLayer, ReshapeLayer)
from lasagne.nonlinearities import LeakyRectify, tanh

from deconv import Deconv2DLayer

logger = logging.getLogger('BGAN.models.dcgan_28_pub')

DIM_X = 31
DIM_Y = 23
DIM_C = 33
NONLIN = None


def build_generator(input_var=None, dim_z=None, dim_h=None):
    layer = InputLayer(shape=(None, dim_z), input_var=input_var)
    layer = batch_norm(DenseLayer(layer, 1024))
    layer = batch_norm(DenseLayer(layer, 6 * 4 * dim_h * 2))
    layer = ReshapeLayer(layer, ([0], dim_h * 2, 6, 4))
    layer = batch_norm(Deconv2DLayer(layer, dim_h, filter_size=(3, 4), stride=2))
    layer = batch_norm(Deconv2DLayer(layer, dim_h, filter_size=(4, 3), stride=1, pad=1))
    layer = Deconv2DLayer(layer, DIM_C, filter_size=(4, 4), stride=2, pad=(1, 1),
                          nonlinearity=NONLIN)

    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer


def build_discriminator(input_var=None, dim_h=None, use_batch_norm=True,
                        leak=None):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)

    layer = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=input_var)
    layer = Conv2DLayer(layer, dim_h, 5, stride=2, pad=2, nonlinearity=lrelu)
    logger.debug('Discriminator output 1: {}'.format(layer.output_shape))
    layer = bn(Conv2DLayer(layer, dim_h * 2, 5, stride=2, pad=2,
                           nonlinearity=lrelu))
    logger.debug('Discriminator output 2: {}'.format(layer.output_shape))
    layer = bn(Conv2DLayer(layer, dim_h * 4, 5, stride=2, pad=2,
                           nonlinearity=lrelu))
    logger.debug('Discriminator output 3: {}'.format(layer.output_shape))
    layer = DenseLayer(layer, 1, nonlinearity=None)

    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer
