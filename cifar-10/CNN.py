#!/usr/bin/env python3
#-*- coding: utf-8

import sys
import os

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, 'models/tutorials/image/cifar10/'))

import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

EPOCHES = 3000
BATCH_SIZE = 128
DATA_DIR = '/tmp/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, wl):
    variable = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(variable), wl)
        tf.add_to_collection('losses', weight_loss)
    return variable


def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == '__main__':
    cifar10.maybe_download_and_extract()
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    images_test, labels_test =cifar10_input.inputs(eval_data=True, data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    images_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 24, 24, 3])
    labels_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE])

    with tf.variable_scope('conv_1'):
        weight_1 = variable_with_weight_loss([5, 5, 3, 64], stddev=0.05, wl=None)
        kernel_1 = tf.nn.conv2d(images_placeholder, weight_1, strides=[1, 1, 1, 1], padding='SAME')
        bias_1 = tf.Variable(tf.zeros([64]))
        conv_1 = tf.nn.relu(kernel_1 + bias_1)
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope('conv_2'):
        weight_2 = variable_with_weight_loss([5, 5, 64, 64], stddev=0.05, wl=None)
        kernel_2 = tf.nn.conv2d(norm_1, weight_2, strides=[1, 1, 1, 1], padding='SAME')
        bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
        conv_2 = tf.nn.relu(kernel_2 + bias_2)
        norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('fc_1'):
        reshape = tf.reshape(pool_2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weight_3 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
        bias_3 = tf.Variable(tf.constant(0.1, shape=[384]))
        fc_1 = tf.nn.relu(tf.matmul(reshape, weight_3) + bias_3)

    with tf.variable_scope('fc_2'):
        weight_4 = variable_with_weight_loss([384, 192], stddev=0.04, wl=0.04)
        bias_3 = tf.Variable(tf.constant(0.1, shape=[192]))
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weight_4) + bias_3)

    with tf.variable_scope('fc_3'):
        weight_5 = variable_with_weight_loss([192, 10], stddev = 1 / 192.0, wl=None)
        bias_5 = tf.Variable(tf.constant(0.0, shape=[10]))
        logits = tf.matmul(fc_2, weight_5) + bias_5

    with tf.variable_scope('loss'):
        loss = loss(logits, labels_placeholder)

    with tf.variable_scope('trainer'):
        trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    for epoch in range(EPOCHES):
        start_time = time.time()
        image_batch, label_batch = session.run([images_train, labels_train])
        _, loss_value = session.run([trainer, loss], feed_dict={images_placeholder: image_batch, labels_placeholder: label_batch})
        duration = time.time() - start_time

        if epoch % 10 == 0:
            examples_per_sec = BATCH_SIZE / duration
            seconds_per_batch = float(duration)

            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (epoch, loss_value, examples_per_sec, seconds_per_batch))

    num_examples = 10000
    num_iter = int(math.ceil(num_examples / BATCH_SIZE))
    top_k = tf.nn.in_top_k(logits, labels_placeholder, 1)
    total_sample_count = num_iter * BATCH_SIZE
    true_count = 0
    while num_iter > 0:
        image_batch, label_batch = session.run([images_test, labels_test])
        predictions = session.run([top_k], feed_dict={images_placeholder: image_batch, labels_placeholder: label_batch})
        true_count += np.sum(predictions)
        num_iter -= 1
    
    precision = true_count / (total_sample_count)
    print('precision @ 1 = %3.f' % precision)