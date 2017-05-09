#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils

def simple (X, classes=2):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    with tf.name_scope('simple'):
        net = slim.conv2d(net, 64, 3, 1)
        net = slim.conv2d(net, 64, 3, 2)

        net = slim.conv2d(net, 128, 3, 1)
        net = slim.conv2d(net, 128, 3, 2)

        net = slim.conv2d(net, 256, 3, 1)
        net = slim.conv2d(net, 256, 3, 2)

        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)

        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)

        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)

        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)

        net = slim.conv2d(net, classes, 1, 1, activation_fn=None, normalizer_fn=None)
        net = slim.conv2d_transpose(net1, num_classes, 17, 8, scope='upscale1')
    net = tf.identity(net, 'logits')
    return net, 8

