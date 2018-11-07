#!/bin/bash
sudo python data/prepare_gtdepth_data.py --dataset_dir=raw_data_NYU/ --dataset_name="kitti_raw_eigen" \
--dump_root=resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4 
