#!/usr/bin/python27
#coding:utf-8

from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel,delayed
import os

parser=argparse.ArgumentParser()
parser.add_argument("--dataset_dir",type=str,default='raw_data_KITTI',help="where the dataset is stored")
parser.add_argument("--dataset_name",type=str,default='kitti_raw_eigen',choices=["kitti_raw_eigen","kitti_raw_stereo","kitti_odom","cityscapes"])
parser.add_argument("--dump_root",type=str,default='resulting/formatted/data',help="where to dump the data")
parser.add_argument("--seq_length",type=int,default=3,help="length of each training sequence")
parser.add_argument("--img_height",type=int,default=128,help="image height")
parser.add_argument("--img_width",type=int,default=416,help="image width")
parser.add_argument("--num_threads",type=int,default=4,help="number of threads to use")
args = parser.parse_args()



