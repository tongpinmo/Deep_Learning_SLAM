#!/usr/bin/python
#coding:utf-8
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice
#这种情况下只是画出了[tx,ty,tz]

# path = 'KITTI/odometry/color/sequences/09/09_groundtruth_full.txt'
# path = 'RGBD/rgbd_dataset_freiburg1_360/pose_groundtruth.txt'
path = 'trajectory.txt'
# path = 'trajectory_c++.txt'


tx = []
ty = []
tz = []

with open(path,'r') as f:

    lines = f.readlines()

    for line in lines:

        pose_tx = line.strip().split(' ')[1]
        pose_ty = line.strip().split(' ')[2]
        pose_tz = line.strip().split(' ')[3]

        tx.append(pose_tx)
        ty.append(pose_ty)
        tz.append(pose_tz)
    # print(tx)

    tx = map(float,tx)
    ty = map(float,ty)
    tz = map(float,tz)

    # print('after map tx:',tx)

    figure = plt.figure()

    ax = figure.add_subplot(111, projection='3d')
    ax.plot(tx,ty,tz)

    # ax = figure.add_subplot(111)
    # ax.plot(tx, ty)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis("equal")
    plt.show()






