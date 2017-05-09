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
    # -50 from L
    # sub-sample AB to 1/4 size
    net = X
    with tf.name_scope('simple'):
        net = slim.conv2d(net, 64, 3, 1)    # conv1_1
        net = slim.conv2d(net, 64, 3, 2)    # conv1_2   -> add bnorm
        net = slim.batch_norm(net)

        # conv2
        net = slim.conv2d(net, 128, 3, 1)
        net = slim.conv2d(net, 128, 3, 2)
        net = slim.batch_norm(net)

        # conv3
        net = slim.conv2d(net, 256, 3, 1) 
        net = slim.conv2d(net, 256, 3, 1)
        net = slim.conv2d(net, 256, 3, 2)
        net = slim.batch_norm(net)

        # conv4
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.batch_norm(net)

        # conv5
        net = slim.conv2d(net, 512, 3, 1)   # dilation 2
        net = slim.conv2d(net, 512, 3, 1)   # dilation 2
        net = slim.conv2d(net, 512, 3, 1)   # conv5_3   dilation 2
        net = slim.batch_norm(net)

        # conv6
        net = slim.conv2d(net, 512, 3, 1)   # dil
        net = slim.conv2d(net, 512, 3, 1)   # dil
        net = slim.conv2d(net, 512, 3, 1)   # dil
        net = slim.batch_norm(net)
        
        # conv7
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)

        # deconv
        net = slim.conv2d_transpose(net, 256, 4, 2)
        net = slim.conv2d(net, 256, 3, 1)
        net = slim.conv2d(net, 256, 3, 1)

        net = slim.conv2d(net, classes, 1, 1, activation_fn=None, normalizer_fn=None)
        # 1/4 size
    net = tf.identity(net, 'logits')
    return net, 8, 4

