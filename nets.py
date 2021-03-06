#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.python.ops import variable_scope

def G (X, channels=1, scope=None, reuse=True):
    assert scope
    net = X
    stack = []
    with variable_scope.variable_scope(scope, None, [net], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                                padding='SAME'):
            stack.append(net)
            net = slim.batch_norm(slim.conv2d(net, 16, 5, 2))
            stack.append(net)       # 1/2
            net = slim.batch_norm(slim.conv2d(net, 32, 3, 1))
            net = slim.max_pool2d(net, 2, 2)
            stack.append(net)       # 1/4
            net = slim.batch_norm(slim.conv2d(net, 64, 3, 1))
            net = slim.max_pool2d(net, 2, 2)
            stack.append(net)       # 1/8
            net = slim.batch_norm(slim.conv2d(net, 128, 3, 1))
            net = slim.max_pool2d(net, 2, 2)
                                    # 1/16
            net = slim.batch_norm(slim.conv2d(net, 128, 3, 1))
            net = slim.batch_norm(slim.conv2d(net, 128, 3, 1))
            net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
                                    # 1/8
            net = tf.concat(3, [net, stack.pop()])
            net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
                                    # 1/4
            net = tf.concat(3, [net, stack.pop()])
            net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
            net = tf.concat(3, [net, stack.pop()])
            net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
            net = tf.concat(3, [net, stack.pop()])
            net = slim.batch_norm(slim.conv2d(net, 32, 5, 1)) 
            net = slim.conv2d(net, channels, 1, 1, activation_fn=None) 
    return tf.identity(net)

def D (X, scope=None, reuse=True):
    net = X
    with variable_scope.variable_scope(scope, None, [net], reuse=reuse):
        net = slim.batch_norm(slim.conv2d(net, 16, 5, 2))
        net = slim.batch_norm(slim.conv2d(net, 32, 3, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 64, 3, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 128, 3, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 256, 3, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 256, 3, 1))
        net = slim.max_pool2d(net, 2, 1)
        net = slim.batch_norm(slim.conv2d(net, 64, 3, 1))
        net = slim.batch_norm(slim.conv2d(net, 32, 3, 1))
        net = slim.conv2d(net, 2, 1, 1, activation_fn=None)
    return net

def resnet (X, scope=None, reuse=True):
    print("USING RESNET")
    net, _ = resnet_v1.resnet_v1_50(X,
                            num_classes=2,
                            global_pool = True,
                            scope=scope,
                            reuse=reuse)
    return net

def resnet_tiny (X, scope=None, reuse=True):
    blocks = [ 
        resnet_utils.Block('block1', resnet_v1.bottleneck,
                           [(64, 32, 1)] + [(64, 32, 2)]),
        resnet_utils.Block('block2', resnet_v1.bottleneck,
                           [(128, 64, 1)] + [(128, 64, 2)]),
        resnet_utils.Block('block3', resnet_v1.bottleneck,
                           [(256, 64, 1)]  + [(128, 64, 2)]),
        resnet_utils.Block('block4', resnet_v1.bottleneck,
                           [(256, 64, 1)]  + [(128, 64, 2)]),
        resnet_utils.Block('block5', resnet_v1.bottleneck,
                           [(256, 64, 1)]  + [(128, 64, 2)]),
        resnet_utils.Block('block6', resnet_v1.bottleneck, [(128, 64, 1)])
    	]   
    net,_ = resnet_v1.resnet_v1(
        X, blocks,
        # all parameters below can be passed to resnet_v1.resnet_v1_??
        num_classes = 2,       # don't produce final prediction
        global_pool = True,       # produce 1x1 output, equivalent to input of a FC layer
        reuse=reuse,              # do not re-use network
        scope=scope)
    return net

