import numpy as np
from config import *
import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import tensorflow as tf

def gg():
    for i in range(5):
        yield 5
        print(i)

for i in gg():
    pass
