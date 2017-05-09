#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import threading
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import picpac
import colorize_nets
import _pic2pic

AB_BINS = 313

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', None, '')
flags.DEFINE_string('net', 'simple', '') 
flags.DEFINE_float('learning_rate', 0.02/100, 'initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_string('model', 'model', '')
flags.DEFINE_string('resume', None, '')

flags.DEFINE_integer('max_steps', 400000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 50, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('max_to_keep', 200, '')

def colorize_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, AB_BINS))
    labels = tf.reshape(labels, (-1, AB_BINS))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    xe = tf.reduce_mean(xe, name='xe')
    loss = xe
    return loss, [xe]

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    X  = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="L")	 # channel L as input
    Y = tf.placeholder(tf.float32, shape=(None, None, None, AB_BINS), name="ab") # soft-binned ab as output

    queue = tf.FIFOQueue(128, (tf.float32, tf.float32))
    enc_XY = queue.enqueue((X, Y))
    dec_XY = queue.dequeue()


    rate = tf.constant(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(rate)

    logits, stride = getattr(colorize_nets, FLAGS.net)(X, classes=AB_BINS)
    #phases = build_graph(A, B, optimizer, optimizer, global_step)
    loss, metrics = colorize_loss(logits, Y)

    train_op = optimizer.minimize(loss, global_step=global_step)

    #metric_names = []
    #for _, _, metrics, _, _ in phases:
    #    metric_names.extend([x.name[:-2] for x in metrics])
    metric_names = [x.name[:-2] for x in metrics]
    #for x in metrics:
    #    tf.summary.scalar(x.name.replace(':', '_'), x)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    init = tf.global_variables_initializer()

    tf.get_default_graph().finalize()

    picpac_config = dict(seed=2016,
                shuffle=True,
                reshuffle=True,
                batch=1,
                split=1,
                split_fold=0,
                round_div=stride,
                channels=3,
                stratify=False,
                pert_min_scale=0.9,
                pert_max_scale=1.5,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )

    stream = picpac.ImageStream(FLAGS.db, perturb=True, loop=True, **picpac_config)

    sess_config = tf.ConfigProto()

    with tf.Session(config=sess_config) as sess:
        coord = tf.train.Coordinator()
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)

        def encode_sample (): 
            while not coord.should_stop():
                images, _, _ = stream.next()
                l, ab, w = _pic2pic.encode_lab(images)
                sess.run([enc_XY], feed_dict={X: l, Y: ab})
            pass

        # create encoding threads
        threads = [threading.Thread(target=encode_sample, args=()) for _ in range(1)]
        for t in threads:
            t.start()

        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            m_avg = np.array([0] * len(metric_names), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                if coord.should_stop():
                    break
                _, m= sess.run([train_op, metrics])
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
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

