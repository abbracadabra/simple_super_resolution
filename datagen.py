import numpy as np
from config import *
import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter

def gendata():
    flist = os.listdir(traindir)
    np.random.shuffle(flist)
    cursor = 0
    while cursor < len(flist):
        im = Image.open(os.path.join(traindir, flist[cursor]))
        w, h = im.size
        i1 = im.crop((0,0,200,200))
        i2 = im.crop((w-200, 0, w, 200))
        i3 = im.crop((w - 200, h-200, w, h))
        i4 = im.crop((0, h - 200, 200, h))
        i1s = i1.filter(ImageFilter.BLUR).resize((100,100))
        i2s = i2.filter(ImageFilter.BLUR).resize((100, 100))
        i3s = i3.filter(ImageFilter.BLUR).resize((100, 100))
        i4s = i4.filter(ImageFilter.BLUR).resize((100, 100))
        hrs  = np.float32([i1,i2,i3,i4])/255.
        lrs = np.float32([i1s,i2s,i3s,i4s])/255.
        yield hrs,lrs




