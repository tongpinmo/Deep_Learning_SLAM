# !/usr/bin/python27
# coding:utf-8

from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length",3, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

#将三张图片连接起来
def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        # print('img_file:',img_file)
        curr_img = scipy.misc.imread(img_file)

        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width)) #调整图像尺寸

        if o == -half_offset:
            image_seq = curr_img
            # print('image_seq.0',image_seq.shape)            # ('image_seq.0', (128, 416, 3))
        else:
            image_seq = np.hstack((image_seq, curr_img))
            # print('image_seq:',image_seq.shape)
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    sfm = SfMLearner()                  #__init__
    sfm.setup_inference(FLAGS.img_height,
                        FLAGS.img_width,
                        'pose',
                        FLAGS.seq_length)
    saver = tf.train.Saver([var for var in tf.trainable_variables()]) 

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
    print('test_frames:',test_frames)
    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()   #list
    times = np.array([float(s[:-1]) for s in times])
    max_src_offset = (FLAGS.seq_length - 1)//2  #1
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):     #tgt_idx=0跳出当前循环
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq = load_image_sequence(FLAGS.dataset_dir, 
                                            test_frames, 
                                            tgt_idx, 
                                            FLAGS.seq_length, 
                                            FLAGS.img_height, 
                                            FLAGS.img_width)
            # print('image_seq.shape',image_seq.shape)                      #(128, 1248, 3)
            # print('image_seq:',image_seq)
            # 传入data,feed_dict={}
            pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')       #an dictionary
            # print('pred_poses.array:',pred['pose'])
            # print('pred_poses.array:', pred['pose'].shape)                  #shape(1,2,6)
            pred_poses = pred['pose'][0]   # dictionary to ndarray
            # print('pred_poses.shape:',pred_poses.shape)                   #shape(2,6)
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)  #FIXME：此处insert zeros,当前帧为基准
            # print('pred_poses',pred_poses)
            # print('pred_poses[0]:',pred_poses[0])
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]    #每张图片对应三帧的时间戳
            # print(type(curr_times))
            out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_poses, curr_times)



main()





