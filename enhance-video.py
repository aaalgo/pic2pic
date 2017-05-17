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
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import cv2

AB_BINS = 313

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('output', None, '')
flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('blur', 1, '')

def main (_):
    logging.basicConfig()

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
        cap = cv2.VideoCapture(FLAGS.input)

        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(FLAGS.output, fourcc, 25, (640*2, 360))
        C = 0
        while cap.isOpened():
            print('%f' % (C/25,))
            ret, frame = cap.read()
            orig = frame.astype(np.float32)
            ks = FLAGS.blur * 2 + 1
            frame = cv2.GaussianBlur(orig, (ks, ks), FLAGS.blur)
            frame = np.expand_dims(frame, 0)
            frame, = sess.run([Y], feed_dict={X: frame})
            both = np.concatenate((orig, frame[0]), axis=1)
            out.write(both.astype(np.uint8))
            C += 1
    pass

if __name__ == '__main__':
    tf.app.run()

