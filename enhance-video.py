#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('build/lib.linux-x86_64-2.7')
sys.path.append('tensorflow-vgg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import threading
import subprocess
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import picpac
import colorize_nets
from vgg19_bgr255 import Vgg19

AB_BINS = 313

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('output', None, '')
flags.DEFINE_string('model', None, '')

def main (_):
    logging.basicConfig(level=FLAGS.verbose)

    X = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    mg = meta_graph.read_meta_graph_file(FLAGS.model + '.meta')
    Y, = tf.import_graph_def(mg.graph_def, name='enhance',
                        input_map={'lo_res:0':X},
                        return_elements=['hi_res:0'])
    saver = tf.train.Saver(saver_def=mg.saver_def, name='enhance')

    init = tf.global_variables_initializer()

    sess_config = tf.ConfigProto()

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        saver.restore(sess, FLAGS.model)
    pass

if __name__ == '__main__':
    tf.app.run()

