#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def maybe_download(file_name, expected_bytes):
    url = 'http://mattmahoney.net/dc/'

    if not os.path.exists(file_name):
        file_name, _ = urllib.request.urlretrieve(url + file_name, file_name)
    
    stat_info = os.stat(file_name)
    if stat_info.st_size == expected_bytes:
        print('found and verified', file_name)
    else:
        print(stat_info.st_size)
        raise Exception('failed to verify ' + file_name)
    
    return file_name


def read_data(file_name):
    with zipfile.ZipFile(file_name) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


VOCABULARY_SIZE = 50000


def build_dataset(words):
    count = [['UNKNOWN', -1]]
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    data = []
    unknown_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unknown_count += 1
        data.append(index)
    
    count[0][1] = unknown_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


DATA = None
DATA_INDEX = 0


def generate_batch(batch_size, num_skips, skip_window):
    global DATA_INDEX

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(DATA[DATA_INDEX])
        DATA_INDEX = (DATA_INDEX + 1) % len(DATA)
    
    # batch_size // num_skips 为本batch中要生成样本的目标单词数
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(DATA[DATA_INDEX])
        DATA_INDEX = (DATA_INDEX + 1) % len(DATA)
    return batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'more labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)


if __name__ == '__main__':
    file_name = maybe_download('text8.zip', 31344016)
    words = read_data(file_name)
    data, count, dictionary, reversed_dictionary = build_dataset(words)
    del words

    DATA = data

    # 每个batch的样本
    batch_size = 128
    # 单词向量维度
    embedding_size = 128
    # 采样窗口大小
    skip_window = 1
    # 每个单词生成的样本数
    num_skips = 2

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # 训练时做负样本的噪声单词数量
    num_sampled = 64

    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])
        # 用于检验的embeddings
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform(shape=[VOCABULARY_SIZE, embedding_size], minval=-1.0, maxval=1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal(
                    shape=[VOCABULARY_SIZE, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)))

            nce_biases = tf.Variable(tf.zeros(shape=[VOCABULARY_SIZE]))

        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=VOCABULARY_SIZE))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        
        # ???
        similarity = tf.matmul(
            valid_embeddings,
            normalized_embeddings,
            transpose_b=True)

        init = tf.global_variables_initializer()

        num_steps = 100001
        with tf.Session(graph=graph) as session:
            init.run()

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = generate_batch(
                    batch_size,
                    num_skips,
                    skip_window)
                feed_dict = {
                    train_inputs: batch_inputs,
                    train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                        print('average loss at step', step, ": ", average_loss)
                        average_loss = 0
                
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = reversed_dictionary[valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                        log_str = 'nearest to %s:' % valid_word

                        for k in range(top_k):
                            close_word = reversed_dictionary[nearest[k]]
                            log_str = '%s %s, ' % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 100
            low_dim_embs = tsne.fit_transform(final_embeddings[: plot_only])
            labels = [reversed_dictionary[i] for i in range(plot_only)]
            plot_with_labels(low_dim_embs, labels)