# author: wangguibo <borgwang@126.com>
# date: 2017-8-28
# Copyright (c) 2017 by wangguibo
#
# file: preprocess.py
# desc: Prepare dataset for predicting task
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np

tf.set_random_seed(666)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore annoying warning of TF


def leaky_relu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


class NeuralNet(object):

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hypers = {
            'learning_rate': 0.003, 'hidden_units': 128, 'batch_size': 128,
            'max_iters': 100000, 'eval_iters': 1000}

        self._contruct_model()
        self._init_sess()

    def _contruct_model(self):
        self.input_data = tf.placeholder(
            shape=[None, self.in_dim], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, ], dtype=tf.int32)
        self.one_hot_labels = tf.one_hot(
            indices=self.labels, depth=2, on_value=1.0, off_value=0.0)
        h1 = tcl.fully_connected(
            inputs=self.input_data,
            num_outputs=self.hypers['hidden_units'],
            weights_regularizer=tcl.l2_regularizer(0.01),
            activation_fn=leaky_relu)
        h2 = tcl.fully_connected(
            inputs=h1, num_outputs=self.hypers['hidden_units'],
            weights_regularizer=tcl.l2_regularizer(0.01),
            activation_fn=leaky_relu)
        self.logits = tcl.fully_connected(
            inputs=h2, num_outputs=self.out_dim,
            weights_regularizer=tcl.l2_regularizer(0.01),
            activation_fn=tf.nn.sigmoid)
        self.pred = tf.argmax(self.logits, 1)
        self.acc = tf.metrics.accuracy(self.labels, self.pred)
        self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.one_hot_labels, logits=self.logits))

        optimizer = tf.train.RMSPropOptimizer(
            self.hypers['learning_rate'], 0.9)
        self.train_op = optimizer.minimize(self.loss)

    def _init_sess(self):
        self.sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        self.sess.run(init)

    def fit(self, x, y):
        running_loss = None

        for i in range(self.hypers['max_iters']):
            idx = np.random.randint(0, len(x), self.hypers['batch_size'])
            batch_x, batch_y = x[idx], y[idx]
            feed_dict = {self.input_data: batch_x, self.labels: batch_y}
            l, _ = self.sess.run([self.loss, self.train_op], feed_dict)
            if running_loss:
                running_loss = 0.99 * running_loss + 0.01 * l
            else:
                running_loss = l
            if i % self.hypers['eval_iters'] == 0:
                print('iter: %d running_loss: %.4f' % (i, running_loss))

    def score(self, x, y):
        feed_dict = {self.input_data: x, self.labels: y}
        acc = self.sess.run(self.acc, feed_dict)[1]
        return acc
