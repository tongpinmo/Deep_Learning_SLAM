#!/usr/bin/python27
#coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from kitti_eval.pose_evaluation_utils import *


# path = 'NYU_testing_KITTI_Pose_output'
# path = 'TUM_testing_KITTI_Pose_output'
path = '../SfMlearner/KITTI_testing_Pose_output'

files = os.listdir(path)
# print('files_length:',len(files))


#将Poses(tx,ty,tz,qx,qy,qz,qw)转化成4*4的变换矩阵T
def TUM_vec_to_Tmat(R,t):

    Tmat = np.concatenate((R,t), axis=1) # shape(3,4)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat


with open('trajectory_SfMlearner.txt','w') as tf:


    for file in files:

        with open(path + '/' + file) as f:
            lines = f.readlines()
            lines = lines[1].strip()
            # print('lines[1]',type(lines))
            time = float(lines.split(' ')[0])

            # time = map(float, time)
            tx = lines.split(' ')[1]
            ty = lines.split(' ')[2]
            tz = lines.split(' ')[3]
            qx = lines.split(' ')[4]
            qy = lines.split(' ')[5]
            qz = lines.split(' ')[6]
            qw = lines.split(' ')[7]
            quat = [qw, qx, qy, qz]
            # print('quat:',quat)
            quat = np.array([float(i) for i in quat])
            R = quat2mat(quat)  # shape(3,3)

            t = [tx, ty, tz]
            t = np.array(map(float, t))
            t = t.reshape(3, 1)

            Tmat = TUM_vec_to_Tmat(R, t)
            # print('Tmat:',Tmat)

        if (file.split('.')[0] == '000000'):
            # print(type(tx))
            tx = float(tx)
            ty = float(ty)
            tz = float(tz)
            qx = float(qx)
            qy = float(qy)
            qz = float(qz)
            qw = float(qw)

            tf.write('%f %f %f %f %f %f %f %f\n' % (0.000000, 0.000000, 0.000000,-0.000000, 0.000000, -0.000000,-0.000000,1.000000))
            tf.write('%f %f %f %f %f %f %f %f\n' % (time, tx, ty, tz, qx, qy, qz, qw))
            this_pose = Tmat

        else:
            this_pose = np.dot(Tmat,this_pose)
            # print('this_pose:',this_pose)
            tx = this_pose[0, 3]
            ty = this_pose[1, 3]
            tz = this_pose[2, 3]
            rot = this_pose[:3, :3]
            qw, qx, qy, qz = rot2quat(rot)
            tf.write('%f %f %f %f %f %f %f %f\n' % (time, tx, ty, tz, qx, qy, qz, qw))







