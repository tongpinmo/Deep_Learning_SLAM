#!/usr/bin/python27
#coding:utf-8

from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import tensorflow as tf
import os

# path = 'raw_data_NYU/Images'
#
#
# def collect_train_frames(path):
#     all_frames = []  #
#     # if os.path.isdir(self.date_list):
#     img_dir = os.path.join(path)
#     N = len(glob(img_dir + '/*.png'))
#     for n in range(1,N):
#         frame_id = '%.d' % n
#         all_frames.append(path + ' ' + frame_id)
#     print all_frames
#
#
# if __name__ == '__main__':
#     collect_train_frames(path)


a = np.zeros((3,3))

ta = tf.convert_to_tensor(a)

print('ta:',ta)

with tf.Session() as sess:
    b = sess.run(ta)
    c = ta.eval()
    print('b:',b)
    print('c:',c)
