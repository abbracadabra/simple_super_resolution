import numpy as np
from config import *
import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import tensorflow as tf
from train import *

im = Image.open(r'D:\githubrepo\simple_super_resolution\outtestimages\wwww.jpg')
im = im.resize((200,200))
im = np.array(im)/255.
im = np.array([im])

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,modelpath)
pp = sess.run(prob,feed_dict={sr:im})
print(pp)


