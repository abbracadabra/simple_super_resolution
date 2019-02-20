import tensorflow as tf
import numpy as np
from transform import *
from discriminate import *
from datagen import *

lr = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
gnet = transformnet(lr)
sr = gnet.output
sr.set_shape([None,200,200,3])
hr = tf.placeholder(dtype=tf.float32,shape=[None,200,200,3])
dnet = discriminatenet(sr)
prob = dnet.output
flag = tf.placeholder(dtype=tf.float32,shape=[None])#0:fake 1:real
dnet_loss = tf.reduce_mean(flag*(1-prob)**2 + (1-flag)*(prob-0)**2)
gan_loss = tf.reduce_mean((1-prob)**2)
pixel_loss = tf.reduce_mean((sr-hr)**2)
gnet_loss = pixel_loss + gan_loss
gnetops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gnet_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gnet'))
dnetops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(dnet_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='dnet'))

with tf.variable_scope('gnet'):
    tf.summary.scalar('pixel_loss', pixel_loss)
    tf.summary.scalar('gan_loss', gan_loss)
    tf.summary.scalar('gnet_loss', gnet_loss)
with tf.variable_scope('dnet'):
    tf.summary.histogram('prob', prob)
    tf.summary.scalar('dnet_loss', dnet_loss)
log_gnet = tf.summary.merge_all(scope='gnet')
log_dnet = tf.summary.merge_all(scope='dnet')
writer = tf.summary.FileWriter(os.path.join(basedir,'log'),graph=tf.get_default_graph())

class Trainer:
    def __init__(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        #self.saver.restore(self.sess,modelpath)
        self.rec = []
        self.minloss = 999999999
        self.minpos = -1
    def train(self):
        for i in range(config.epochs):
            for i,(hrs, lrs) in enumerate(gendata()):
                _, self.glossval, fakeimgs, log1 = self.sess.run([gnetops, gnet_loss, sr, log_gnet], feed_dict={hr: hrs, lr: lrs})
                mixims = np.concatenate((hrs, fakeimgs))
                flgs = np.array([1] * len(hrs) + [0] * len(fakeimgs))
                _, dlossval, log2 = self.sess.run([dnetops, dnet_loss, log_dnet], feed_dict={sr: mixims, flag: flgs})
                writer.add_summary(log1)
                writer.add_summary(log2)
                self.savebest()
                self.preview()
    def savebest(self):
        print(self.glossval)
        if len(self.rec) - self.minpos > 100:
            raise Exception("performance not improving")
        self.rec.append(self.glossval)
        if self.glossval < self.minloss:
            self.minloss = self.glossval
            self.minpos = len(self.rec) - 1
            if len(self.rec)%1 ==0:
                self.saver.save(self.sess, config.modelpath)
    def preview(self):
        if len(self.rec)%1==0:
            flist = os.listdir(testdir)
            fn = np.random.choice(flist)
            fp = os.path.join(testdir, fn)
            im = Image.open(fp)
            im = np.array(im) / 255.
            fakeim = self.sess.run(sr, feed_dict={lr: np.array([im])})
            im = Image.fromarray(np.squeeze(np.uint8(fakeim * 255)))
            im.save(os.path.join(outtestdir, fn))

if __name__ == '__main__':
    Trainer().train()








