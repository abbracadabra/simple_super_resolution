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
flag = tf.placeholder(dtype=tf.float32,shape=[None])#0:fake 1:real
dnet_loss = flag*(1-prob)**2 + (1-flag)(prob-0)**2
gan_loss = (1-prob)**2
pixel_loss = (sr-hr)**2/tf.size(hr)
gnet_loss = pixel_loss + gan_loss
gnetops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gnet_loss,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gnet'))
dnetops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(dnet_loss,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='dnet'))

class Trainer:
    def __init__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.rec = []
        self.minloss = 999999999
        self.minpos = -1
    def train(self):
        for i in range(config.epochs):
            for hrs, lrs in gendata():
                _, self.glossval, fakeimgs = self.sess.run([gnetops, gnet_loss, sr], feed_dict={hr: hrs, lr: lrs})
                mixims = np.concatenate((hrs, fakeimgs))
                flgs = np.array([1] * len(hrs) + [0] * len(fakeimgs))
                _, dlossval = self.sess.run([dnetops, dnet_loss], feed_dict={sr: mixims, flag: flgs})
                self.callback()
    def callback(self):
        if len(self.rec) - self.minpos > 50:
            raise Exception("performance not improving")
        self.rec.append(self.glossval)
        if self.glossval < self.minloss:
            self.minloss = self.glossval
            self.minpos = len(self.rec) - 1
            if len(self.rec) % 20:
                self.saver.save(self.sess, config.modelpath)


if __name__ == '__main__':
    Trainer().train()








