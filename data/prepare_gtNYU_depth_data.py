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
parser.add_argument("--dataset_dir", type=str, default='raw_data_NYU', help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, default='kitti_raw_eigen', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom"])
parser.add_argument("--dump_root", type=str, default='resulting/formatted/data/ ', help="Where to dump the data")
parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):  # i是序号，im是下标对应的数据
        if i == 0:
            res = im
        else:
            res = np.hstack((res,im))
    return res

# 读入raw_data_KITTI文件夹数据，格式化，写入相机内参
def dump_example(n):
    print('n:',n)
    if n % 200 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)  #{'file_name':, 'image_seq': }的一个字典
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])

    dump_dir = os.path.join(args.dump_root, example['folder_name'])  #dump_root=resulting/formatted/data/ --seq_length=3
    # print('dump_dir:',dump_dir)
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    # formatted之后的文件，_*.jpg ,路径在/resulting/formatted/data/下面
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))


def main():
    if not os.path.exists(args.dump_root):
        print(args.dump_root)
        os.makedirs(args.dump_root)

    global data_loader  #为一个定义在函数外的变量赋值，该变量名是全局的

    if args.dataset_name == 'kitti_raw_eigen':   #执行这个
        from kitti.kitti_raw_gtdepth_loader import kitti_raw_gtdepth_loader
        data_loader = kitti_raw_gtdepth_loader(args.dataset_dir,
                                                img_height=args.img_height,
                                                img_width=args.img_width,
                                                seq_length=args.seq_length)



    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n) for n in range(data_loader.num_train))

    # Split into depth
    np.random.seed(8964)
    folders = os.listdir(args.dump_root + 'Depth')
    # print('subfolders:',folders)
    # 路径/resulting /formatted /data/
    with open(args.dump_root + 'depth.txt', 'w') as tf:

        imfiles = glob(os.path.join(args.dump_root,'Depth','*.jpg'))
        imfiles = natsort.natsorted(imfiles)
        # print('imfiles:',imfiles)
        frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
        print('frame_ids:',frame_ids)
        for frame in frame_ids:
            tf.write('%s %s\n' % ('Depth', frame))


main()




