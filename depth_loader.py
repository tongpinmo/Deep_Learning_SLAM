#!/usr/bin/python27
#coding:utf-8
from __future__ import division
import os
import random
import tensorflow as tf


class DepthLoader(object):
    #_init_是python class中的构造函数,初始化实例的值
    def __init__(self,
                 dataset_dir='resulting/formatted/data_NYU',
                 # dataset_dir='resulting/formatted/data_TUM',
                 batch_size = 4,
                 img_height = 128,
                 img_width  = 416,
                 num_source = 2,
                 num_scales = 4):

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_depth_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)  #integer creating in a particular scope
        # Load the list of training files into queues

        file_list = self.format_file_list(self.dataset_dir, 'depth') #读取resulting/formatted/data_/  下面的depth.txt文件
        # print('depth_file_list:',file_list)
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'],
            seed=seed,
            shuffle=True)   #True stands for random sequences
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader() #reader
        _, image_contents = img_reader.read(image_paths_queue) #reader读取序列
        image_seq = tf.image.decode_jpeg(image_contents)       #解码，tensor
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # print('tgt_image.shape',tgt_image.shape)                # shape=(128,416,1)
        # print('src_image_stack',src_image_stack.shape)          # shape=(128,416,2)

        # Form training batches
        src_image_stack, tgt_image = \
                tf.train.batch([src_image_stack, tgt_image],
                               batch_size=self.batch_size)

        # print('tgt_image.shape', tgt_image.shape)                       #shape(4,128,416,1)
        # print('src_image_stack', src_image_stack.shape)                 #shape(4,128,416,2)

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)      #shape(4,128,416,3)
        # print('image_all:',image_all)
        image_all = self.data_augmentation(
            image_all, self.img_height, self.img_width)
        # print('image_all:',image_all)

        tgt_image = image_all[:, :, :, :1]                               #shape(4,128,416,1)
        # print('tgt_image.shape:',tgt_image.shape)
        src_image_stack = image_all[:, :, :, 1:]                         #shape(4,128,416,2)
        # print('src_image_stack.shape:', src_image_stack.shape)


        return tgt_image, src_image_stack

    def data_augmentation(self, im, out_h, out_w):
        # Random scaling
        def random_scaling(im):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)                   #shape(2,),范围在(1,1.5)之间
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])               #此处将image尺寸改变了

            return im

        # Random cropping　随机剪切
        def random_cropping(im, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            # in_h = tf.Print(in_h,[in_h],message='in_h')                 #FIXME：此处输入的image大小已经发生变化
            # in_w = tf.Print(in_w, [in_w], message='in_w')
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            # offset_y = tf.Print(offset_y, [offset_y], message='offset_y:')
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            #FIXME：offset_y,offset_x 不是[0,1]之间的数
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w) # FIXME:图像的裁剪,图像的左上角位于offset_height,offset_width,中心改变

            return im
        im = random_scaling(im)
        im = random_cropping(im, out_h, out_w)
        im = tf.cast(im, dtype=tf.float32)
        return im

#all_list 包括.jpg
    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f: #打开 resulting/formatted/data/depth.txt
             frames=f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]          #[:-1]操作去掉'\n'
        # print('frame_ids;',frame_ids)
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        return all_list

#取出image_sequence
    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))   #416
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # tgt_image = tf.Print(tgt_image,[tgt_image],message='tgt_image',summarize=20)
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])          #shape(?,416,?)
        # src_image_1 = tf.Print(src_image_1,[src_image_1.shape],message='src_image_1.shape')
        # print('src_image_1.shape',src_image_1.shape)
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # src_image_2=tf.Print(src_image_2,[src_image_2.shape[0]],message='src_image_2.shape')                         #shape(?,416,?)
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)   #(,416*2,)
        # print('src_image_seq.shape',src_image_seq.shape)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)   #在axis=2 即color channels上concat
        # print('src_image_stack.shape',src_image_stack.shape)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 1])  # 此处乘以３是因为每张图片都是[image_height,image_width,1],在axis=2上连接，就是6,因此 shape(128,416,2)
        # print('src_image_stack.set_shape', src_image_stack.shape)     #(128,416,6)
        tgt_image.set_shape([img_height, img_width, 1])   #shape(128,416,3)
        return tgt_image, src_image_stack

#FIXME:
    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,  # shape(1,128,416,3)
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,  # shape(1,128,832,3)
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,  # # shape(1,128,832,3)
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)           #  shape(1,128,1664,3)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,                    #   shape(1,128,416,6)
                                    [0, 0, i*img_width, 0],
                                    [-1, -1, img_width, -1])
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack





