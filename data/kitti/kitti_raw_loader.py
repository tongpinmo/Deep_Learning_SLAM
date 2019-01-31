#!/usr/bin/pythohn27
#coding:utf-8

from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc

class kitti_raw_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split,
                 img_height=256,
                 img_width=256,
                 seq_length=3):
        dir_path = os.path.dirname(os.path.realpath(__file__))      #返回kitti_raw_loader.py文件执行的绝对路径

        self.dataset_dir = dataset_dir   #raw_data_NYU
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.date_list = ['Images']
        self.collect_train_frames()

#将raw_data中的Images 中的数据格式变为 formatted/data里面的命名格式
    def collect_train_frames(self):
        all_frames = []  #
        # if os.path.isdir(self.date_list):
        for date in self.date_list:
            img_dir = os.path.join(self.dataset_dir + date)
            print('img_dir:',img_dir)

            N = len(glob(img_dir + '/*.png'))
            for n in range(1,N+1):
                frame_id = '%.d' % n
                all_frames.append(date + ' ' + frame_id)
        # print("all_frames",all_frames)

        self.train_frames = all_frames
        # print('self.train_frames:',self.train_frames)
        # print('type of self.train_frames:', type(self.train_frames))
        self.num_train = len(self.train_frames)                 #2284
        # print('self.num_train:',self.num_train)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive,_ = frames[tgt_idx].split(' ')
        # print('tgt_drive:',tgt_drive)
        half_offset = int((self.seq_length - 1)/2)             #1
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        # print('min_src_drive:',min_src_drive)
        max_src_drive, _ = frames[max_src_idx].split(' ')
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
        # print('tgt_idx:',tgt_idx)
        for o in range(-half_offset, half_offset + 1): #range(-1,2)
            curr_idx = tgt_idx + o
            # print('curr_idx:',curr_idx)
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            # print('curr_drive:', curr_drive, 'curr_frame_id:', curr_frame_id)
            curr_img = self.load_image_raw(curr_drive, curr_frame_id)
            # print curr_img.shape      #(480, 640, 3)
            if o == 0:  #FIXME：求出缩放比例，后面对intrinsics使用
                zoom_y = self.img_height/curr_img.shape[0]  #128.0/480=0.266666667
                # print('zoom_y',zoom_y)
                zoom_x = self.img_width/curr_img.shape[1]   #416.0/640=0.65
                # print('zoom_x', zoom_x)
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))  #将原始图片resize为128*416
            image_seq.append(curr_img)  #shape(128,416,3)
        #     print('image_seq:',image_seq)
        # print('image_seq:',len(image_seq))
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        # print('tgt_idx:',tgt_idx)
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        # print('image_seq:', len(image_seq))       #3
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        # print('tgt_drive',tgt_drive,'tgt_frame_id',tgt_frame_id)
        intrinsics = self.load_intrinsics_raw(tgt_drive, tgt_frame_id)   #读取 calib_cam_to_cam.txt
        print('intrinsics before scale',intrinsics)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)     # 经过了尺度变换,放入._cam.txt中
        print('intrinsics after scale', intrinsics)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, drive, frame_id):

        img_file = os.path.join(self.dataset_dir, drive,frame_id + '.png')
        # print('img_file:',img_file)
        img = scipy.misc.imread(img_file)   #将图像转换为ndarray数组
        # print('img:',img)
        # print('img.shape:',img.shape)           #shape(480,640,3)
        return img

    def load_intrinsics_raw(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir,'calib_cam_to_cam.txt')      #:'raw_data_NYU/calib_cam_to_cam.txt'
        # print('calib_file',calib_file)

        filedata = self.read_raw_calib_file(calib_file)
        # print('filedata',filedata)
        P_rect = np.reshape(filedata['P_rect'], (3, 4))
        # print('cid',cid)
        # print('P_rect',P_rect)
        intrinsics = P_rect[:3, :3]
        # print('intrinsics from calib_txt',intrinsics)
        return intrinsics

    def read_raw_calib_file(self,filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data

#FIXME:此处对内参矩阵做出了相应地缩放
    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out


