import numpy as np
from config import *
import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import tensorflow as tf

im = Image.open(r'D:\githubrepo\simple_super_resolution\outtestimages\000010.jpg')
im = np.array(im)/255.
im = np.array([im])

saver = tf.train.import_meta_graph(modelpath+".meta")
sess = tf.Session()
saver.restore(sess,modelpath)

input = tf.get_default_graph().get_tensor_by_name("input:0")
output = tf.get_default_graph().get_tensor_by_name("output:0")
sr = sess.run(output,feed_dict={input:im})
sr = np.squeeze(np.uint8(sr*255))
Image.fromarray(sr).show()