#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import picpac
import nets

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('A', None, '')
flags.DEFINE_string('B', None, '')
flags.DEFINE_string('G', 'G', '')
flags.DEFINE_string('D', 'D', '')
flags.DEFINE_string('opt', 'adam', '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_float('momentum', 0.99, 'when opt==mom')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 200, '')
flags.DEFINE_string('log', None, 'tensorboard')
flags.DEFINE_integer('max_summary_images', 20, '')
flags.DEFINE_integer('channels', 1, '')
flags.DEFINE_integer('stride', 32, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('max_to_keep', 1000, '')

def LossG (X, Y, name):
    diff = tf.subtract(X, Y)
    L = tf.sqrt(tf.reduce_mean(tf.multiply(diff, diff)))
    return tf.identity(L, name=name)

def LossD (logits, l, name):
    logits = tf.reshape(logits, (-1, 2))

    shape = tf.unpack(tf.shape(logits))
    shape.pop()
    shape = tf.pack(shape)
    if l == 0:
        labels = tf.zeros(shape, tf.int32)
    else:
        labels = tf.ones(shape, tf.int32)
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    xe = tf.reduce_mean(xe)
    return tf.identity(xe, name=name)

def build_graph (A, B, optimizer, global_step):

    G = getattr(nets, FLAGS.G)  # generator generator
    D = getattr(nets, FLAGS.D)  # discriminator generator

    #   A  -->  aB  --> abA, which must be same as A
    aB = G(A, scope='G/ab', reuse=True)
    abA = G(aB, scope='G/ba', reuse=True)
    #   B  -->  bA  --> baB, which must be same as B
    bA = G(B, scope='G/ba', reuse=True)
    baB = G(bA, scope='G/ab', reuse=True)

    a1 = D(A, scope='D/a')
    abL = D(aB, scope='D/b')
    
    b1 = D(B, scope='D/b')
    baL = D(bA, scope='D/a')

    phases = []

    # phase one
    l1 = LossG(abA, A, 'Gaba')
    l2 = LossG(baB, B, 'Gbab')
    l3 = LossD(abL, 1, 'Gab')
    l4 = LossD(baL, 1, 'Gba')
    loss = l1 + l2 + l3 + l4

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "G")
    phases.append(('generate',
                  optimizer.minimize(loss, global_step=global_step, var_list=var_list),
                  [l1, l2, l3, l4],  # metrics
                  [bA, aB]))

    l1 = LossD(a1, 1, 'Da1')
    l2 = LossD(bA, 0, 'Da0')
    l3 = LossD(b1, 1, 'Db1')
    l4 = LossD(aB, 0, 'Db0')
    loss = l1 + l2 + l3 + l4

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D")
    phases.append(('discriminate',
                  optimizer.minimize(loss, global_step=global_step, var_list=var_list),
                  [l1, l2, l3, l4],
                  []))
    return phases

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    A = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="A")
    B = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="B")

    rate = FLAGS.learning_rate
    if FLAGS.opt == 'adam':
        rate /= 100
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', rate)
    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(rate)
    elif FLAGS.opt == 'mom':
        optimizer = tf.train.MomentumOptimizer(rate, FLAGS.momentum)
    else:
        optimizer = tf.train.GradientDescentOptimizer(rate)
        pass

    phases = build_graph(A, B, optimizer, global_step)

    metric_names = []
    for train, metrics in phases:
        metric_names.extend([x.name[:-2] for x in metrics])

    if FLAGS.log:
        train_summaries = tf.summary.merge_all()
        assert not train_summaries is None
        summary_writer = tf.summary.FileWriter(FLAGS.log, tf.get_default_graph(), flush_secs=20)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    tf.get_default_graph().finialize()

    picpac_config = dict(seed=2016,
                shuffle=True,
                reshuffle=True,
                batch=1,
                split=1,
                split_fold=0,
                round_div=FLAGS.stride,
                channels=FLAGS.channels,
                stratify=True,
                pert_color1=20,
                pert_angle=20,
                pert_min_scale=0.9,
                pert_max_scale=1.5,
                pert_hflip=True,
                pert_vflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )

    streamA = picpac.ImageStream(FLAGS.A, perturb=True, loop=True, **picpac_config)
    streamB = picpac.ImageStream(FLAGS.B, perturb=True, loop=True, **picpac_config)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            avg = np.array([0] * len(metric_names), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                a, _, _ = streamA.next()
                b, _, _ = streamB.next()
                forward_dict = {}
                m_off = 0
                for _, train, metrics, forward in phases:
                    feed_dict = {A: a, B: b}
                    feed_dict.update(forward_dict)
                    _, m, f, _ = sess.run([train, metrics, forward, train_summaries], feed_dict=feed_dict)
                    forward_dict = dict(zip(forward, f))
                    m_off_n = m_off + len(metrics)
                    avg[m_off:m_off_n] += m
                    m_off = m_off_n
                    pass
                step += 1
                pass
            avg /= FLAGS.epoch_steps
            stop_time = time.time()
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step %d: elapsed=%.4f time=%.4f, %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            if summary_writer:
                summary_writer.add_summary(summaries, step)
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                start_time = time.time()
                saver.save(sess, ckpt_path)
                stop_time = time.time()
                print('epoch %d step %d, saving to %s in %.4fs.' % (epoch, step, ckpt_path, stop_time - start_time))
            pass
        pass
    if summary_writer:
        summary_writer.close()
    pass

if __name__ == '__main__':
    tf.app.run()

