#!/bin/bash
sudo python train.py --dataset_dir=resulting/formatted/data/ --checkpoint_dir=checkpoints/ \
--img_width=416 --img_height=128 --batch_size=4
