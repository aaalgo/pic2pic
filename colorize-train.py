#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('build/lib.linux-x86_64-2.7')
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import threading
import subprocess
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2
import picpac
import colorize_nets
import _pic2pic

AB_BINS = 313

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'ilsvrc2015.train', '')
flags.DEFINE_string('net', 'simple', '') 
flags.DEFINE_float('learning_rate', 0.02/100, 'initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_string('model', 'model', '')
flags.DEFINE_string('log', 'log', '')
flags.DEFINE_string('resume', None, '')

flags.DEFINE_integer('batch', 48, '')
flags.DEFINE_integer('max_steps', 400000, '')
flags.DEFINE_integer('epoch_steps', 200, '')
flags.DEFINE_integer('ckpt_epochs', 20, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('max_to_keep', 200, '')
flags.DEFINE_integer('encoding_threads', 4, '')

VIS_KEY = 'visualize'

# weighted cross entropy
def colorize_loss (logits, labels, weights):
    # to HWC
    logits = tf.reshape(logits, (-1, AB_BINS))
    labels = tf.reshape(labels, (-1, AB_BINS))
    weights = tf.reshape(weights, (-1,))
    xe = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    xe = tf.reduce_mean(xe * weights, name='xe')
    tf.summary.scalar('xe', xe)
    loss = xe
    return loss, [xe]

ab_dict = _pic2pic.ab_dict()

def decode_lab (l, ab):
    ab_flat = np.reshape(ab, (-1, ab.shape[-1]))
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
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    X  = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="input")
    Y  = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="output")

    L  = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="L")	 # channel L as input
    AB = tf.placeholder(tf.float32, shape=(None, None, None, AB_BINS), name="ab") # soft-binned ab as output
    W  = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="W")	 # channel L as input


    queue = tf.FIFOQueue(128, (tf.float32, tf.float32, tf.float32))
    enc = queue.enqueue((L, AB, W))
    dec_L, dec_AB, dec_W = queue.dequeue()
    dec_L.set_shape(L.get_shape())
    dec_AB.set_shape(AB.get_shape())
    dec_W.set_shape(W.get_shape())
    tf.summary.image('input', X, max_outputs=5)
    tf.summary.image('output', Y, max_outputs=5)

    rate = tf.constant(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', rate)
    optimizer = tf.train.AdamOptimizer(rate)

    logits, stride, downsize = getattr(colorize_nets, FLAGS.net)(dec_L, classes=AB_BINS)
    prob = tf.nn.softmax(logits)

    loss, metrics = colorize_loss(logits, dec_AB, dec_W)

    train_op = optimizer.minimize(loss, global_step=global_step)

    metric_names = [x.name[:-2] for x in metrics]

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    summaries = tf.summary.merge_all()
    if FLAGS.resume is None:
        if FLAGS.log[0] != '/':
            subprocess.check_call("rm -rf %s" % FLAGS.log, shell=True)
    log = tf.summary.FileWriter(FLAGS.log, tf.get_default_graph(), flush_secs=20)

    init = tf.global_variables_initializer()

    tf.get_default_graph().finalize()

    picpac_config = dict(seed=2016,
                cache=False,
                max_size=200,
                min_size=192,
                crop_width=176,
                crop_height=176,
                shuffle=True,
                reshuffle=True,
                batch=FLAGS.batch,
                round_div=stride,
                channels=3,
                stratify=False,
                pert_min_scale=1, #0.92,
                pert_max_scale=1.5,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )

    stream = picpac.ImageStream(FLAGS.db, perturb=False, loop=True, **picpac_config)

    sess_config = tf.ConfigProto()

    with tf.Session(config=sess_config) as sess:
        coord = tf.train.Coordinator()
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)

        def encode_sample (): 
            while not coord.should_stop():
                images, _, _ = stream.next()
                l, ab, w = _pic2pic.encode_lab(images, downsize)
                sess.run([enc], feed_dict={L: l, AB: ab, W: w})
            pass

        # create encoding threads
        threads = [threading.Thread(target=encode_sample, args=()) for _ in range(FLAGS.encoding_threads)]
        for t in threads:
            t.start()

        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            m_avg = np.array([0] * len(metric_names), dtype=np.float32)
            for i in tqdm(range(FLAGS.epoch_steps), leave=False):
                if coord.should_stop():
                    break
                if i + 1 == FLAGS.epoch_steps:
                    # run with summary
                    l, ab, w = sess.run([dec_L, dec_AB, dec_W])
                    x = decode_lab(l, ab)
                    ab_p, = sess.run([prob], feed_dict={dec_L: l, dec_AB: ab, dec_W: w})
                    y = decode_lab(l, ab_p)
                    _, m, s = sess.run([train_op, metrics, summaries], feed_dict={dec_L: l, dec_AB: ab, X: x, Y: y})
                    log.add_summary(s, step)
                else:
                    _, m = sess.run([train_op, metrics])
                m_avg += m
                step += 1
                pass

            m_avg /= FLAGS.epoch_steps
            stop_time = time.time()

            epoch += 1
            saved = ''
            if epoch % FLAGS.ckpt_epochs == 0:
                saver.save(sess, '%s/%d' % (FLAGS.model, step))
                saved = ' saved'

            m_txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(m_avg))])
            print('step=%d elapsed=%.4f/%.4f %s%s'
                    % (step, (stop_time - start_time), (stop_time - global_start_time), m_txt, saved))
            pass
        coord.request_stop()
        coord.join(threads)
        log.close()
        pass
    
    pass


if __name__ == '__main__':
    tf.app.run()

