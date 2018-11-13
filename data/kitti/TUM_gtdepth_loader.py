#!/usr/bin/pythohn27
#coding:utf-8

from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc

class TUM_gtdepth_loader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=256,
                 img_width=256,
                 seq_length=5):
        dir_path = os.path.dirname(os.path.realpath(__file__))   #

        self.dataset_dir = dataset_dir   ##RGBD/rgbd_dataset_freiburg1_360
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.date_path = os.path.join(self.dataset_dir + '/' + 'depth.txt')
        self.collect_train_frames()

#将raw_data中的Depth 中的数据格式变为 formatted/data里面的命名格式
    def collect_train_frames(self):
        all_frames = []  #
        # if os.path.isdir(self.date_list):
        with open(self.date_path, 'r') as f:
            lines = f.readlines()
            all_frames = lines
        # print('all_frames:',all_frames)

        self.train_frames = all_frames
        # print('self.train_frames:',self.train_frames)
        # print('type of self.train_frames:',type(self.train_frames))
        self.num_train = len(self.train_frames)  # 774
        # print('self.num_train:',self.num_train)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive,_ = frames[tgt_idx].split('/')
        # print('tgt_drive:',tgt_drive)
        half_offset = int((self.seq_length - 1)/2)             #1
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split('/')
        # print('min_src_drive:',min_src_drive)
        max_src_drive, _ = frames[max_src_idx].split('/')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive :
            return True
        return False

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        # print('example:',example)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):

        ## FIXME:resize
        half_offset = int((seq_length - 1)/2) #1
        image_seq = []
        for o in range(-half_offset, half_offset + 1): #range(-1,2)
            curr_idx = tgt_idx + o
            curr_drive    = frames[curr_idx].strip().split('/')[0]
            # print('curr_drive:',curr_drive)
            curr_frame_id = frames[curr_idx].strip().split('/')[1][:-4]
            # print('curr_frame_id:', curr_frame_id)
            curr_img = self.load_image_raw(curr_drive, curr_frame_id)
            # print curr_img.shape      #(480, 640)
            if o == 0:  #FIXME：求出缩放比例，后面对intrinsics使用
                zoom_y = self.img_height/curr_img.shape[0]  #128.0/480=0.266666667
                # print('zoom_y',zoom_y)
                zoom_x = self.img_width/curr_img.shape[1]   #416.0/640=0.65
                # print('zoom_x', zoom_x)
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))  #将原始图片resize为128*416
            # print('shape of curr_img:',curr_img.shape)             #shape(128,416)
            image_seq.append(curr_img)
            # print('image_seq:',image_seq)
        return image_seq

    def load_example(self, frames, tgt_idx):
        image_seq = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive    = frames[tgt_idx].strip().split('/')[0]
        tgt_frame_id = frames[tgt_idx].strip().split('/')[1][:-4]
        # print('tgt_drive',tgt_drive,'tgt_frame_id',tgt_frame_id)
        example = {}
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, drive, frame_id):

        img_file = os.path.join(self.dataset_dir, drive,frame_id + '.png')
        # print('img_file:',img_file)
        img = scipy.misc.imread(img_file)   #将图像转换为ndarray数组
        # print('img:',img)
        # img = img.reshape(480,640,1)
        # print('img.shape:',img.shape)           #shape(480,640)
        return img


