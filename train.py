import tensorflow as tf
import numpy as np
from transform import *
from discriminate import *
from datagen import *

lr = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
gnet = transformnet(lr)
sr = gnet.output
hr = tf.placeholder(dtype=tf.float32,shape=[None,200,200,3])
dnet = discriminatenet(sr)
prob = dnet.output
flag = tf.placeholder(dtype=tf.float32,shape=[1])#0:fake 1:real
dnet_loss = flag*(1-prob)**2 + (1-flag)(prob-0)**2
gan_loss = (1-prob)**2
pixel_loss = (sr-hr)**2/tf.size(hr)
gnet_loss = pixel_loss + gan_loss
tf.train.AdamOptimizer(learning_rate=0.001).minimize(gnet_loss,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gnet'))
tf.train.AdamOptimizer(learning_rate=0.001).minimize(dnet_loss,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='dnet'))

def train():
    sess = tf.Session()
    for i in range(config.epochs):
        for lr, hr in gendata():
            pass

if __name__ == '__main__':
    train()








