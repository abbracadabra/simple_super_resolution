import tensorflow as tf
import numpy as np

with tf.variable_scope("ssx"):
    aa = tf.placeholder(dtype=tf.float32,shape=[None,10,10,3])
    bb = tf.layers.conv2d(aa,3,3)
    cc = tf.layers.conv2d(bb, 3, 3)

print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='sxsx'))
