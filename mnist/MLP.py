#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    session = tf.InteractiveSession()

    n_input = 784
    n_hidden = 300

    W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1))
    b1 = tf.Variable(tf.zeros([n_hidden]))
    W2 = tf.Variable(tf.zeros([n_hidden, 10]))
    b2 = tf.Variable(tf.zeros([10]))

    x = tf.placeholder(tf.float32, shape=[None, n_input])
    keep_prob = tf.placeholder(tf.float32)

    hidden = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
    hidden_drop = tf.nn.dropout(hidden, keep_prob)
    y = tf.nn.softmax(tf.add(tf.matmul(hidden_drop, W2), b2))

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    trainer = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    tf.global_variables_initializer().run()
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        trainer.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

