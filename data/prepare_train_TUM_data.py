#!/usr/bin/python27
#coding:utf-8

from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os
import natsort

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='RGBD/rgbd_dataset_freiburg1_360', help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, default='kitti_raw_TUM', choices=["kitti_raw_eigen", "kitti_raw_TUM", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root", type=str, default='resulting/formatted/data_TUM/ ', help="Where to dump the data")
parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

#FIXME：将三帧连起来
def concat_image_seq(seq):
    for i, im in enumerate(seq):  # i是索引，im是索引对应的数据
        if i == 0:
            res = im
        else:
            res = np.hstack((res,im))
    return res

# 读入raw_data_KITTI文件夹数据，格式化，写入相机内参
def dump_example(n):
    if n % 100 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)  #{'file_name':, 'image_seq': }的一个字典
    # print('example:',example)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    # print('image_seq.shape:',image_seq.shape)
    intrinsics = example['intrinsics']  #此处的相机内参矩阵已经经过了尺度变换
    #相机内参，３*３的矩阵
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])  #dump_root=resulting/formatted/data/ --seq_length=3
    # print('dump_dir:',dump_dir)
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    # formatted之后的文件，_jpg+_cam.txt ,路径在/resulting/formatted/data/下面
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        #写入相机内参矩阵
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

def main():
    if not os.path.exists(args.dump_root):
        print(args.dump_root)
        os.makedirs(args.dump_root)

    global data_loader  #为一个定义在函数外的变量赋值，该变量名是全局的

    if args.dataset_name == 'kitti_raw_TUM':   #执行这个
        from kitti.TUM_raw_loader import TUM_raw_loader
        data_loader = TUM_raw_loader(args.dataset_dir,
                                     img_height=args.img_height,
                                     img_width=args.img_width,
                                     seq_length=args.seq_length)



    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n) for n in range(data_loader.num_train))

    # Split into train/val
    np.random.seed(8964)
    # subfolders = os.listdir(args.dump_root)
    # print('subfolders:',subfolders)             #['rgb']
    # 路径/resulting /formatted /data/
    with open(args.dump_root + 'train.txt', 'w') as tf:
        imfiles = glob(os.path.join(args.dump_root, 'rgb','*.jpg'))
        imfiles = natsort.natsorted(imfiles)
        # print('imfiles:',imfiles)

        frame_ids = [os.path.basename(fi)[:-4] for fi in imfiles]
        for frame in frame_ids:
            tf.write('%s %s\n' % ('rgb', frame))

main()




