#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from skimage import measure
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac
import _pic2pic

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', None, '')
flags.DEFINE_string('output', None, '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_integer('stride', 8, '')
flags.DEFINE_integer('max', 32, '')
flags.DEFINE_float('T', None, '')

ab_dict = _pic2pic.ab_dict()

def decode_lab (l, ab, T=None):
    ab_flat = np.reshape(ab, (-1, ab.shape[-1]))
    if not T is None:   # need to test
        o = ab_flat
        ab_flat += 1e-15
        np.log(ab_flat, ab_flat)
        ab_flat /= T
        np.exp(ab_flat, ab_flat)
        S = np.sum(ab_flat, axis=0)
        assert S.shape[0] == ab_flat.shape[0]
        ab_flat /= S
        assert np.byte_bounds(o) == np.byte_bounds(ab_flat)
        pass

    ab_small = np.reshape(np.dot(ab_flat, ab_dict), ab.shape[:3] + (2,))

    _, H, W, _ = l.shape
    lab_one = np.zeros((H, W, 3), dtype=np.float32)

    rgb = np.zeros(l.shape[:3] + (3,), dtype=np.float32)
    for i in range(l.shape[0]):
        lab_one[:, :, :1] = l[i]
        lab_one[:, :, 1:] = cv2.resize(ab_small[i], (W, H))
        rgb[i] = cv2.cvtColor(lab_one, cv2.COLOR_LAB2RGB)
        pass
    return rgb

def main (_):
    assert FLAGS.db and os.path.exists(FLAGS.db)
    assert FLAGS.model and os.path.exists(FLAGS.model + '.meta')

    L = tf.placeholder(tf.float32, shape=(None, None, None, 1))

    mg = meta_graph.read_meta_graph_file(FLAGS.model + '.meta')
    logits, = tf.import_graph_def(mg.graph_def, name='colorize',
                        input_map={'L:0':X},
                        return_elements=['logits:0'])
    saver = tf.train.Saver(saver_def=mg.saver_def, name='colorize')

    picpac_config = dict(seed=2016,
                cache=False,
                max_size=200,
                min_size=192,
                shuffle=True,
                reshuffle=True,
                batch=1,
                round_div=stride,
                channels=3,
                stratify=False,
                pert_min_scale=1, #0.92,
                pert_max_scale=1.5,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )

    stream = picpac.ImageStream(FLAGS.db, perturb=False, loop=False, **picpac_config)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, FLAGS.model)
        y, = sess.run([Y], feed_dict={X: batch})
        gallery = Gallery(FLAGS.out, cols=2, header=['groundtruth', 'prediction'])
        c = 0
        for images, _, _ in stream:
            if FLAGS.max and (c >= FLAGS.max):
                break
            l, ab, w = _pic2pic.encode_lab(images, downsize)
            ab_p, = sess.run([logits], feed_dict={L: l})
            y = decode_lab(l, ab_p, T=FLAGS.T)
            cv2.imwrite(gallery.next(), images[0])
            cv2.imwrite(gallery.next(), y[0])
            pass
        gallery.flush()
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

