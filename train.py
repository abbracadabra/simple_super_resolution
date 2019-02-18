import tensorflow as tf
import numpy as np
from transform import *
from discriminate import *

gnet = transformnet()
sr = gnet.output
hr = tf.placeholder(dtype=tf.float32,shape=[None,200,200,3])
dnet1 = discriminatenet(sr)
dnet2 = discriminatenet(hr)




