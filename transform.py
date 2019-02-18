import tensorflow as tf
import numpy as np
import config



class transformnet:
    def __init__(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        tmp = self.buildresblock(self.input,64)
        tmp = self.buildresblock(tmp, 64)
        tmp = self.buildresblock(tmp, 64)
        tmp = self.buildupsample(tmp)
        tmp = self.buildconv(tmp,3,1)
        self.output = tmp
    def buildresblock(self,input,filters):
        tmp = tf.layers.conv2d(input,filters,3,padding='SAME',use_bias=False)
        tmp = tf.layers.batch_normalization(tmp)
        tmp = tf.contrib.layers.bias_add(tmp)
        tmp = tf.add(tmp,input)
        tmp = tf.nn.leaky_relu(tmp)
        return tmp
    def buildupsample(self,input,filters):
        tmp = tf.layers.conv2d_transpose(input, filters, 3, 2, 'SAME', use_bias=False)
        tmp = tf.layers.batch_normalization(tmp)
        tmp = tf.contrib.layers.bias_add(tmp)
        tmp = tf.nn.leaky_relu(tmp)
        return tmp
    def buildconv(self,input,filters,kernal):
        tmp = tf.layers.conv2d(input, filters, kernal, padding='SAME', use_bias=False)
        tmp = tf.layers.batch_normalization(tmp)
        tmp = tf.contrib.layers.bias_add(tmp)
        tmp = tf.nn.leaky_relu(tmp)
        return tmp




