import numpy as np
from config import *
import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter

im = Image.open(r"D:\Users\yl_gong\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000015.jpg")
imm = im.filter(ImageFilter.BLUR)
imm.save(r"D:\Users\yl_gong\Desktop\wwww.jpg")

