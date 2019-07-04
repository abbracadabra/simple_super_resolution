import os

basedir = os.path.dirname(os.path.realpath(__file__))
epochs = 3
traindir = r'D:\Users\xxx\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
modelpath = os.path.join(basedir,"mdl","srgan")
testdir = os.path.join(basedir,"testimages")
outtestdir = os.path.join(basedir,"outtestimages")


