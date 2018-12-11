#!/bin/bash
sudo python train.py --dataset_dir=resulting/formatted/data_NYU/ --checkpoint_dir=checkpoints_NYU/ \
--img_width=416 --img_height=128 --batch_size=4
