#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import math
import time
import sys
import tensorflow as tf

from datetime import datetime


def network(images):
    with tf.name_scope('conv1') as scope:
        weights1 = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=0.1), name='weights')
        conv1 = tf.nn.conv2d(images, weights1, strides=[1, 4, 4, 1], padding='SAME')
        biases1 = tf.Variable(tf.zeros([64]), trainable=True, name='biases')
        kernel1 = tf.nn.relu(conv1 + biases1)
        lrn1 = tf.nn.lrn(kernel1, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn')
        max_pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool')

    with tf.name_scope('conv2') as scope:
        weights2 = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=0.1), name='weights')
        conv2 = tf.nn.conv2d(max_pool1, weights2, strides=[1, 1, 1, 1],  padding='SAME')
        biases2 = tf.Variable(tf.zeros([192]), trainable=True, name='biases')
        kernel2 = tf.nn.relu(conv2 + biases2)
        lrn2 = tf.nn.lrn(kernel2, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn')
        max_pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool')

    with tf.name_scope('conv3') as scope:
        weights3 = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=0.1), name='weights')
        conv3 = tf.nn.conv2d(max_pool2, weights3, strides=[1, 1, 1, 1], padding='SAME')
        biases3 = tf.Variable(tf.zeros([384]), trainable=True, name='biases')
        kernel3 = tf.nn.relu(conv3 + biases3)

    with tf.name_scope('conv4') as scope:
        weights4 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.1), name='weights')
        conv4 = tf.nn.conv2d(kernel3, weights4, strides=[1, 1, 1, 1], padding='SAME')
        biases4 = tf.Variable(tf.zeros([256]), trainable=True, name='biases')
        kernel4 = tf.nn.relu(conv4 + biases4)

    with tf.name_scope('conv5') as scope:
        weights5 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=0.1), name='weights')
        conv5 = tf.nn.conv2d(kernel4, weights5, strides=[1, 1, 1, 1], padding='SAME')
        biases5 = tf.Variable(tf.zeros([256]), trainable=True, name='biases')
        kernel5 = tf.nn.relu(conv5 + biases5)
        max_pool5 = tf.nn.max_pool(kernel5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool')

    with tf.name_scope('fc1') as scope:
        dim = 13 * 13 * 256
        flat = tf.reshape(max_pool5, shape=[-1, dim])
        weights_fc1 = tf.Variable(tf.truncated_normal([dim, 4096], dtype=tf.float32, stddev=0.1), name='weights')
        biases_fc1 = tf.Variable(tf.zeros([4096]), trainable=True, name='biases')
        fc_1 = tf.nn.relu(tf.matmul(flat, weights_fc1) + biases_fc1)
    
    with tf.name_scope('fc2') as scope:
        weights_fc2 = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=0.1), name='weights')
        biases_fc2 = tf.Variable(tf.zeros([4096]), trainable=True, name='biases')
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights_fc2) + biases_fc2)
        
    
