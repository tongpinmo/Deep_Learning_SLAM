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


path = 'RGBD/rgbd_dataset_freiburg1_360'

with open(path + '/' + 'associate.txt') as f:
    with open(path + '/' + 'depth.txt','w') as df:
        with open(path + '/' + 'rgb.txt','w') as rf:
            for lines in f.readlines():
                lines = lines.split(' ')
                # print('lines:',lines)
                depth = lines[1]
                print('depth:',depth)
                rgb   = lines[3]
                print('rgb:',rgb)
                df.write('%s\n' % (depth))
                rf.write('%s' % (rgb)  )
























