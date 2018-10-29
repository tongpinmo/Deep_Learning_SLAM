#!/usr/bin/python27
#coding:utf-8
#pylab inline
from __future__ import division
import matplotlib
matplotlib.use('TkAgg')  # matplotlib 'agg'是不画图的，'Tkagg'是画图的．
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from SfMLearner import SfMLearner
from utils import normalize_depth_for_display
import matplotlib.pyplot as plt
import operator

img_height=128
img_width=416

# ckpt_file = 'models/model-190532'  #depth
ckpt_file='checkpoints/model-117270'
fh = open('misc/sample.png', 'r')
#

# fh=open('raw_data_KITTI/2011_09_28/2011_09_28_drive_0001_sync/image_02/data/0000000012.png','r') # 自己测试所用
I = pil.open(fh) #读取图片
I = I.resize((img_width, img_height), pil.ANTIALIAS)  #antialias滤镜缩放
I = np.array(I)
print(I.shape)   #(128, 416, 3)
print(I[None,:,:,:].shape) #(1,128, 416, 3) 增加一个维度，[batch,img_height,img_width,channels]


sfm = SfMLearner() #initialize
sfm.setup_inference(img_height,
                    img_width,
                    mode='depth')

saver = tf.train.Saver([var for
                        var in tf.model_variables()])  #保存和恢复变量,保存到checkpoints中
with tf.Session() as sess:
    saver.restore(sess, ckpt_file)
    pred = sfm.inference(I[None,:,:,:], sess, mode='depth') #I[None,:,:,:] None的作用是增加了一个轴
    print(pred)  #is a dictionary
    print(pred['depth'][0,:,:,0])
    print(pred['depth'][0,:,:,0].shape)


plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(I)
plt.subplot(1,2,2); plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
plt.show()




