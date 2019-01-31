#!/usr/bin/python27
#coding:utf-8
import os
import PIL.Image as pil
import numpy as np
from depth_loader import DepthLoader
import tensorflow as tf


class Getdepth(object):
    def __init__(self,
                 dataset_dir='resulting/formatted/data_TUM',
                 # dataset_dir='resulting/formatted/data_NYU',
                 batch_size = 4,
                 img_height = 128,
                 img_width  = 416,
                 num_source = 2,
                 num_scales = 4):

        self.dataset_dir = dataset_dir
        print('self.dataset_dir:',self.dataset_dir)
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales


    def get_depth_graph(self):

        loader = DepthLoader(self.dataset_dir,
                             self.batch_size,
                             self.img_height,
                             self.img_width,
                             self.num_source,
                             self.num_scales)

        with tf.name_scope("data_loading"):

            tgt_image, _ = loader.load_depth_batch()
            print('tgt_image:',tgt_image)                               #tensor.shape(4,128,416,1),float32


        return tgt_image




