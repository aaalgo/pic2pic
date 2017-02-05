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
flags.DEFINE_string('Gnet', 'G', '')
flags.DEFINE_string('Dnet', 'resnet_tiny', '')
flags.DEFINE_string('opt', 'adam', '')
flags.DEFINE_float('eta', 0.1, '')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_float('momentum', 0.99, 'when opt==mom')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 400000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 50, '')
flags.DEFINE_string('log', None, 'tensorboard')
flags.DEFINE_integer('max_summary_images', 20, '')
flags.DEFINE_integer('channels', 1, '')
flags.DEFINE_integer('stride', 32, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('max_to_keep', 200, '')

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
    return tf.identity(xe, name=name) #, tf.identity(L,name=name+'H')

def build_graph (A, B, Gopt, Dopt, global_step):

    G = getattr(nets, FLAGS.Gnet)  # generator generator
    D = getattr(nets, FLAGS.Dnet)  # discriminator generator

    #   A  -->  aB  --> abA, which must be same as A
    aB = G(A, channels=FLAGS.channels, scope='G/ab', reuse=False)
    tf.identity(aB, name='AB')
    abA = G(aB, channels=FLAGS.channels, scope='G/ba', reuse=False)
    tf.identity(abA, name='ABA')
    #   B  -->  bA  --> baB, which must be same as B
    bA = G(B, channels=FLAGS.channels, scope='G/ba', reuse=True)
    tf.identity(bA, name='BA')
    baB = G(bA, channels=FLAGS.channels, scope='G/ab', reuse=True)
    tf.identity(baB, name='BAB')

    a1 = D(A, scope='D/a', reuse=False)
    abL = D(aB, scope='D/b', reuse=False)
    
    b1 = D(B, scope='D/b', reuse=True)
    baL = D(bA, scope='D/a', reuse=True)

    phases = []

    # phase one
    l1 = LossG(abA, A, 'Gaba')
    l2 = LossG(baB, B, 'Gbab')
    l3 = LossD(abL, 1, 'Gab')
    l4 = LossD(baL, 1, 'Gba')
    insure_pos_correlation = LossG(aB, A, 'insureAB') + LossG(bA, B, 'insureBA')
    loss = (l1 + l2) * (FLAGS.eta/2) + (l3 + l4)/2
    if FLAGS.resume is None:
        loss = tf.cond(global_step < 1000,
                        lambda: loss + insure_pos_correlation,
                        lambda: loss)

    loss = tf.identity(loss, name='G')

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "G")
    phases.append(('generate',
                  Gopt.minimize(loss, global_step=global_step, var_list=var_list),
                  [loss, l1, l2, tf.identity((l3+l4)/2,name='Gxe')],  # metrics
                  [bA, aB], []))

    l1 = LossD(a1, 1, 'Da1')
    l2 = LossD(baL, 0, 'Da0')
    l3 = LossD(b1, 1, 'Db1')
    l4 = LossD(abL, 0, 'Db0')
    #acc = tf.identity((h1+h2+h3+h4)/4, name='acc')
    loss = tf.identity((l1 + l2 + l3 + l4)/4, name='Dxe')

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D")
    phases.append(('discriminate',
                  Dopt.minimize(loss, var_list=var_list),
                  [loss],
                  [], []))
    vA = tf.saturate_cast(tf.concat(2, [A, abA, aB]), dtype=tf.uint8, name='vA')
    vB = tf.saturate_cast(tf.concat(2, [B, baB, bA]), dtype=tf.uint8, name='vB')

    tf.summary.image('A', vA)
    tf.summary.image('B', vB)
    return phases

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.A and os.path.exists(FLAGS.A)
    assert FLAGS.B and os.path.exists(FLAGS.B)

    A = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="A")
    B = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="B")

    rate = tf.constant(FLAGS.learning_rate)
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

    phases = build_graph(A, B, optimizer, optimizer, global_step)

    metric_names = []
    for _, _, metrics, _, _ in phases:
        metric_names.extend([x.name[:-2] for x in metrics])

    summaries = tf.constant(1)
    summary_writer = None
    if FLAGS.log:
        summaries = tf.summary.merge_all()
        assert not summaries is None
        summary_writer = tf.summary.FileWriter(FLAGS.log, tf.get_default_graph(), flush_secs=20)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    init = tf.global_variables_initializer()
    tf.get_default_graph().finalize()

    graph = tf.get_default_graph()
    graph.finalize()
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
        if node.name[0] == 'D':
            #print(node.name)
            pass

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
        sess.run(init)
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
                for _, train, metrics, forward, show in phases:
                    feed_dict = {A: a, B: b}
                    feed_dict.update(forward_dict)
                    _, m, f, ss= sess.run([train, metrics, forward, show], feed_dict=feed_dict)
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
            print('step %d: elapsed=%.4f time=%.4f %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            if summary_writer:
                s, = sess.run([summaries], feed_dict=feed_dict)
                summary_writer.add_summary(s, step)
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

