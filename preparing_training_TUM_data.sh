#!/bin/bash
sudo python data/prepare_train_TUM_data.py --dataset_dir='RGBD/rgbd_dataset_freiburg1_360' --dataset_name="kitti_raw_TUM" \
--dump_root=resulting/formatted/data_TUM/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4 
