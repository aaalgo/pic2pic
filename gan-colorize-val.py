#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('build/lib.linux-x86_64-2.7')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac
import _pic2pic
from gallery import Gallery

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', None, '')
flags.DEFINE_string('output', None, '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_integer('stride', 8, '')
flags.DEFINE_integer('downsize', 4, 'has no effect')
flags.DEFINE_integer('max', 32, '')
flags.DEFINE_float('T', None, '')
flags.DEFINE_float('s_add', None, '')
flags.DEFINE_float('s_mul', None, '')

def main (_):
    assert FLAGS.db and os.path.exists(FLAGS.db)
    assert FLAGS.model and os.path.exists(FLAGS.model + '.meta')

    GRAY = tf.placeholder(tf.float32, shape=(None, None, None, 1))

    mg = meta_graph.read_meta_graph_file(FLAGS.model + '.meta')
    COLOR, = tf.import_graph_def(mg.graph_def, name='colorize',
                        #input_map={'L:0':L},
                        input_map={'gray:0':GRAY},
                        return_elements=['color:0'])
    #prob = tf.nn.softmax(logits)
    saver = tf.train.Saver(saver_def=mg.saver_def, name='colorize')

    picpac_config = dict(seed=2016,
                cache=False,
                max_size=200,
                min_size=192,
                crop_width=192,
                crop_height=192,
                shuffle=True,
                #reshuffle=True,
                batch=1,
                round_div=FLAGS.stride,
                channels=3,
                stratify=False,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )

    stream = picpac.ImageStream(FLAGS.db, perturb=False, loop=False, **picpac_config)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, FLAGS.model)
        gallery = Gallery(FLAGS.output, cols=2, header=['groundtruth', 'prediction'])
        c = 0
        for images, _, _ in stream:
            if FLAGS.max and (c >= FLAGS.max):
                break
            gray, _, _ = _pic2pic.encode_bgr(images.copy(), FLAGS.downsize)
            #l, ab, w = _pic2pic.encode_lab(images.copy(), FLAGS.downsize)
            #
            color, = sess.run([COLOR], feed_dict={GRAY: gray})

            cv2.imwrite(gallery.next(), gray[0])

            full = np.zeros(images.shape, dtype=np.float32)
            color /= 255.0
            gray /= 255.0
            _, H, W, _ = images.shape
            for i in range(images.shape[0]):
                lab = cv2.cvtColor(cv2.cvtColor(gray[i], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
                print(lab.shape)
                full[i, :, :, :1] = lab[:, :, :1]
                one = cv2.resize(color[i], (W, H))

                lab = cv2.cvtColor(one, cv2.COLOR_BGR2LAB)
                full[i, :, :, 1:] = lab[:, :, 1:]
                cv2.cvtColor(full[i], cv2.COLOR_LAB2BGR, full[i])
                if FLAGS.s_add and FLAGS.s_mul:
                    hsv = cv2.cvtColor(full[i], cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    s *= FLAGS.s_mul
                    s += FLAGS.s_add
                    hsv = cv2.merge([h, s, v])
                    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, full[i])
                pass
            full *= 255
            cv2.imwrite(gallery.next(), full[0])
            #y_p = decode_lab(l, ab_p, T=FLAGS.T)
            c += 1
            print('%d/%d' % (c, FLAGS.max))
            pass
        gallery.flush()
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

