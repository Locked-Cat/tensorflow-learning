#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    session = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    b_conv1 = bias_variable([32], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    b_conv2 = bias_variable([64], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
    b_fc1 = bias_variable([1024], name='b_fc1')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    W_fc2 = weight_variable([1024, 10], name='W_fc2')
    b_fc2 = bias_variable([10], name='b_fc2')
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('cost', cost)
    trainer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    '''
    日了狗，简直蠢到爆炸。。。
    血泪的教训，argmax一定别忘了带个维度的参数！！
    每个batch的shape形如(n, 10)， n是batch的大小，要实现比较的目的话，argmax的维度应该是1
    '''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./CNN_log', session.graph)
    tf.global_variables_initializer().run()
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %f' % (i, train_accuracy))
        summary, _ = session.run([merged, trainer], feed_dict={
                                 x: batch[0], y_: batch[1], keep_prob: 0.5})
        writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    session.close()
    writer.close()