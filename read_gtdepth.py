#!/usr/bin/python27
#coding:utf-8
import os
import PIL.Image as pil
import numpy as np
import natsort
import tensorflow as tf

img_height=128
img_width=416

# path ='/mnt/PI_Lab/users/zhaoyong/Dataset/NYU/NYUv1/Depth'
path = 'raw_data_NYU/Depth'
files = os.listdir(path)
files = natsort.natsorted(files)


def getdepth():

    for file in files:

        fh = open(path+'/'+file, 'r')
        I = pil.open(fh)
        I = I.resize((img_width, img_height), pil.ANTIALIAS)
        I = np.array(I)
        I = I[None,:,:,None]
        I = tf.convert_to_tensor(I,dtype=tf.float32)
        print('the type of I ',I)
        pred_depth = I

    return pred_depth




if __name__ == '__main__':
    pred_detpth = getdepth()
    print('pre_depth:',pred_detpth.shape)

