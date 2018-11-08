#!/bin/bash
python test_kitti_pose.py --test_seq 9 --seq_length 3 --dataset_dir KITTI/odometry/color/ --output_dir KITTI_testing_Pose_output/ --ckpt_file checkpoints/model-19812
