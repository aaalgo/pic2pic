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
import _pic2pic
from vgg19_bgr255 import Vgg19

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', '/data/scratch/wdong/imagenet.db', '')
flags.DEFINE_float('learning_rate', 0.01/100, 'initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_string('model', 'model', '')
flags.DEFINE_string('log', 'log', '')
flags.DEFINE_string('resume', None, '')

flags.DEFINE_integer('batch', 64, '')
flags.DEFINE_integer('max_steps', 80000000, '')
flags.DEFINE_integer('epoch_steps', 200, '')
flags.DEFINE_integer('ckpt_epochs', 20, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('max_to_keep', 200, '')
flags.DEFINE_integer('encoding_threads', 4, '')


flags.DEFINE_integer('discriminator_size', 32, '')
flags.DEFINE_float('perceptual_weight', 1e0, '')
flags.DEFINE_float('smoothness_weight', 2e5, '')
flags.DEFINE_float('adversary_weight', 5e2, '')

def p_relu (alpha=0.25):
    return lambda net: tf.nn.relu(net) - alpha * tf.nn.relu(-net)

def softminus (x):
    return x - tf.nn.softplus(x)

def generator (net, scope='G'):
    with tf.variable_scope(scope):
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
        net = slim.conv2d(net, 512, 3, 1, rate=2)
        net = slim.conv2d(net, 512, 3, 1, rate=2)
        net = slim.conv2d(net, 512, 3, 1, rate=2)
        net = slim.batch_norm(net)
        # conv6
        net = slim.conv2d(net, 512, 3, 1, rate=2)
        net = slim.conv2d(net, 512, 3, 1, rate=2)
        net = slim.conv2d(net, 512, 3, 1, rate=2)
        net = slim.batch_norm(net)
        # conv7
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        net = slim.conv2d(net, 512, 3, 1)
        # deconv
        net = slim.conv2d_transpose(net, 256, 4, 2)
        net = slim.conv2d(net, 256, 3, 1)
        net = slim.conv2d(net, 256, 3, 1)
        net = slim.conv2d(net, 3, 1, 1, activation_fn=None, normalizer_fn=None)
        net = tf.clip_by_value(net, 0, 255)
        # 1/4 size
    # net, stride, down-size
    return net, 8, 4

# returns two feature vectors, L and H
# L is low-level perceptual feature
# H is high-level discriminative feature, +1: real data; -1: generated data
def feature_extractor (bgr255, scope='D'):
    c = FLAGS.discriminator_size
    vgg = Vgg19()   # vgg net is not trainable
    vgg.build(bgr255)
    with tf.variable_scope(scope):
        disc1_1 = slim.conv2d(slim.batch_norm(vgg.conv1_2), c, 5, 2, activation_fn=p_relu())
        disc1_2 = slim.conv2d(disc1_1, c, 5, 2)     # 1/4
        disc2 = slim.conv2d(slim.batch_norm(vgg.conv2_2), 2*c, 5, 2, activation_fn=p_relu())    # 1/4
        disc3 = slim.conv2d(slim.batch_norm(vgg.conv3_2), 3*c, 3, 1, activation_fn=p_relu())    # 1/4
        net = tf.concat([disc1_2, disc2, disc3], axis=3)
        net = slim.conv2d(net, 4*c, 1, 1, activation_fn=p_relu())
        net = slim.conv2d(net, 3*c, 3, 2, activation_fn=p_relu())
        net = slim.conv2d(net, 2*c, 1, 1, activation_fn=p_relu())
        disc = slim.batch_norm(slim.conv2d(net, 1, 1, 1, activation_fn=None))
    return vgg.conv2_2, disc

def BGR2RGB (x):
    chs = tf.unstack(x, axis=-1)
    return tf.stack([chs[2], chs[1], chs[0]], axis=-1)

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    rate = tf.constant(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', rate)
    optimizer = tf.train.AdamOptimizer(rate)

    enc_GRAY  = tf.placeholder(tf.float32, shape=(None, None, None, 1))  # inference input	
    enc_GTCOLOR = tf.placeholder(tf.float32, shape=(None, None, None, 3))  # inference output
    enc_W  = tf.placeholder(tf.float32, shape=(None, None, None, 1))	 # inference per-pixel weight

    queue = tf.FIFOQueue(128, (tf.float32, tf.float32, tf.float32))
    enc = queue.enqueue((enc_GRAY, enc_GTCOLOR, enc_W))
    GRAY, GTCOLOR, W = queue.dequeue()
    GRAY.set_shape(enc_GRAY.get_shape())
    # we need to name this so as to map the variable for prediction
    GRAY = tf.identity(GRAY, name='gray')
    GTCOLOR.set_shape(enc_GTCOLOR.get_shape())
    W.set_shape(enc_W.get_shape())

    COLOR, stride, downsize = generator(GRAY)
    COLOR = tf.identity(COLOR, name='color')
    #tf.summary.image('low_res', BGR2RGB(X), max_outputs=3)
    #tf.summary.image('hi_res', BGR2RGB(Y), max_outputs=3)
    #tf.summary.image('predict', BGR2RGB(G), max_outputs=3)
    tf.summary.image('input', BGR2RGB(GTCOLOR), max_outputs=5)
    tf.summary.image('output', BGR2RGB(COLOR), max_outputs=5)


    # X and G need to go through the same feature extracting network
    # so we concatenate them first and then split the results
    L, H = feature_extractor(tf.concat([GTCOLOR, COLOR], axis=0))

    L1, L2 = tf.split(L, 2)  # low-level feature,    2 is prediction
    H1, H2 = tf.split(H, 2)  # high-level feature,   2 is prediction

    loss_perceptual = tf.reduce_mean(tf.square(L1 - L2), name='pe')

    sub = tf.shape(COLOR) - tf.constant([0, 1, 1, 0], dtype=tf.int32)
    G0 = tf.slice(COLOR, [0, 0, 0, 0], sub)
    Gx = tf.slice(COLOR, [0, 0, 1, 0], sub)
    Gy = tf.slice(COLOR, [0, 1, 0, 0], sub)

    loss_smooth = tf.identity(tf.reduce_mean(
                    tf.pow(tf.square(G0 - Gx) + tf.square(G0 - Gy), 1.25)) / 529357.9139706489,
                    name = 'sm')

    D_real = tf.reduce_mean(tf.nn.softplus(H1), name='dr')
    D_fake = tf.reduce_mean(softminus(H2), name='df')

    loss_adversary = tf.identity(1.0 - D_fake, name='ad')


    loss_G = tf.identity(loss_perceptual * FLAGS.perceptual_weight + 
                         loss_smooth * FLAGS.smoothness_weight + 
                         loss_adversary * FLAGS.adversary_weight, name='lg')

    loss_D = tf.identity(D_fake - D_real, name='ld')

    phases = []
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "G")
    phases.append(('generate',
                  optimizer.minimize(loss_G, global_step=global_step, var_list=var_list),
                  [loss_G, loss_perceptual, loss_smooth, loss_adversary],
                  [GTCOLOR, COLOR]))

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D")
    phases.append(('discriminate',
                  optimizer.minimize(loss_D, global_step=global_step, var_list=var_list),
                  [loss_D, D_real, D_fake],
                  []))

    metric_names = []
    for _, _, metrics, _ in phases:
        metric_names.extend([x.name[:-2] for x in metrics])

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    summaries = tf.summary.merge_all()
    if FLAGS.resume is None:
        if FLAGS.log[0] != '/':
            subprocess.check_call("rm -rf %s" % FLAGS.log, shell=True)
    log = tf.summary.FileWriter(FLAGS.log, tf.get_default_graph(), flush_secs=20)

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
                pert_min_scale=0.97, #0.92,
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
                gray, color, w = _pic2pic.encode_bgr(images, downsize)
                #print("GRAY:", gray.shape)
                #print("COLOR:", color.shape)
                sess.run([enc], feed_dict={enc_GRAY: gray, enc_GTCOLOR: color, enc_W: w})
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
            avg = np.array([0] * len(metric_names), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                forward_dict = {}
                m_off = 0
                for _, train, metrics, forward in phases:
                    feed_dict = {}
                    feed_dict.update(forward_dict)
                    _, m, f = sess.run([train, metrics, forward], feed_dict=feed_dict)
                    forward_dict = dict(zip(forward, f))
                    m_off_n = m_off + len(metrics)
                    avg[m_off:m_off_n] += m
                    m_off = m_off_n
                    pass
                step += 1
                pass
            images, _, _ = stream.next()
            images = images[:5, :, :, :]
            gray, gtcolor, _ = _pic2pic.encode_bgr(images, downsize)
            color = sess.run(COLOR, feed_dict={GRAY: gray})
            full = np.zeros(images.shape, dtype=np.float32)
            _, H, W, _ = images.shape
            color /= 255.0
            gray /= 255.0
            for i in range(images.shape[0]):
                lab = cv2.cvtColor(cv2.cvtColor(gray[i], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
                full[i, :, :, :1] = lab[:, :, :1]
                one = cv2.resize(color[i], (W, H))
                lab = cv2.cvtColor(one, cv2.COLOR_BGR2LAB)
                full[i, :, :, 1:] = lab[:, :, 1:]
                cv2.cvtColor(full[i], cv2.COLOR_LAB2BGR, full[i])
                pass
            s, = sess.run([summaries], feed_dict={COLOR: full, GTCOLOR: images})
            log.add_summary(s, step)
            avg /= FLAGS.epoch_steps
            stop_time = time.time()
            epoch += 1
            saved = ''
            if epoch % FLAGS.ckpt_epochs == 0:
                saver.save(sess, '%s/%d' % (FLAGS.model, step))
                saved = ' saved'

            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step=%d elapsed=%.4f/%.4f  %s%s'
                    % (step, (stop_time - start_time), (stop_time - global_start_time), txt, saved))
            pass
        pass
        coord.request_stop()
        coord.join(threads)
        log.close()
    pass

if __name__ == '__main__':
    tf.app.run()

