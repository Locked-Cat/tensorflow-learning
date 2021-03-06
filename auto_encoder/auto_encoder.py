#!/usr/bin/env python3

import numpy as np
import sklearn.preprocessing as prep 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low

    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer, scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.network_weights = self._initialize_weights()

        # define construction of the network
        self.x = tf.placeholder(tf.float32, shape=[None, n_input])
        self.hidden = self.transfer_function(tf.add(tf.matmul(self.x + self.scale * tf.random_normal((self.n_input,)), self.network_weights['W1']), self.network_weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.network_weights['W2']), self.network_weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['W1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['W2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    def partial_fit(self, X):
        cost, _ = self.session.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.session.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        return self.session.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.network_weights['b1'])
        return self.session.run(self.reconstruction, feed_dict={self.hidden: hidden})
    
    def reconstruct(self, X):
        return self.session.run(self.reconstruction, feed_dict={self.x:X, self.scale: self.training_scale})


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)
    tranining_epochs = 20
    batch_size =128
    display_step = 1

    auto_encoder = AdditiveGaussianNoiseAutoencoder(
        n_input=784,
        n_hidden=200,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
        scale=0.01)

    for epoch in range(tranining_epochs):
        avg_cost = 0.0
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = auto_encoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print('Epoch: %04d' % epoch, 'cost=', '{:.9f}'.format(avg_cost))