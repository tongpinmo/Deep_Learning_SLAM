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



path = 'RGBD/rgbd_dataset_freiburg1_360'

with open(path + '/' + 'associate.txt') as f:
    with open(path + '/' + 'depth.txt','w') as df:
        with open(path + '/' + 'rgb.txt','w') as rf:
            for lines in f.readlines():
                lines = lines.split(' ')
                # print('lines:',lines)
                rgb   = lines[1]
                print('rgb:',rgb)
                depth = lines[3]
                print('depth:', depth)


                rf.write('%s\n' % (rgb))
                df.write('%s' % (depth))
























