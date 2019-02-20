import tensorflow as tf
import numpy as np
import config

class discriminatenet:
    def __init__(self,input):
        with tf.variable_scope('dnet'):
            self.input = input
            tmp = self.buildconv(input, 64, 1)
            tmp = self.buildconv(tmp, 64, 1)
            tmp = self.buildconv(tmp, 128, 2)
            tmp = self.buildconv(tmp, 128, 1)
            tmp = self.buildconv(tmp, 256, 2)
            tmp = self.buildconv(tmp, 256, 1)
            tmp = tf.layers.flatten(tmp)
            tmp = tf.layers.dense(tmp,units=1, activation=tf.nn.sigmoid)
            self.output = tmp
    def buildconv(self,input,filters,stride):
        tmp = tf.layers.conv2d(input, filters, 3, stride, padding='SAME', use_bias=False)
        tmp = tf.layers.batch_normalization(tmp)
        tmp = tf.contrib.layers.bias_add(tmp)
        tmp = tf.nn.leaky_relu(tmp)
        return tmp




