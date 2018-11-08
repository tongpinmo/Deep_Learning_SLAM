#!/usr/bin/python27
#coding:utf-8
import os
import PIL.Image as pil
import numpy as np
from depth_loader import DepthLoader
import tensorflow as tf

img_height=128
img_width=416


class Getdepth(object):
    def __init__(self):
        pass


    def getdepth(self,opt):
        opt.num_source = opt.seq_length - 1  # seq_length=3 已定,source_view=2 and target_view=1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.get_depth_graph()

    def get_depth_graph(self):
        opt = self.opt

        loader = DepthLoader(opt.dataset_dir,
                             opt.batch_size,
                             opt.img_height,
                             opt.img_width,
                             opt.num_source,
                             opt.num_scales)

        with tf.name_scope("data_loading"):
            tgt_image, _ = loader.load_depth_batch()
            print('tgt_image:',tgt_image)
            tgt_image = self.preprocess_image(tgt_image)




        return tgt_image




    def preprocess_image(self,image):
        # Assuming input image is uint8
        # print('image uint8: ', image)
        # image = tf.Print(image, [image], "image uint8: ")
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # 转化为[0,1)的float
        # image = tf.Print(image, [image], "image float32: ")
        return image * 2. - 1.  # 扩大范围,(-1,1)



