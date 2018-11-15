#!/usr/bin/python27
#coding:utf-8
from kitti_eval.pose_evaluation_utils import *
import os
import natsort
import numpy as np
import  matplotlib.pyplot as  plt
from mpl_toolkits.mplot3d import Axes3D

# path = 'NYU_testing_KITTI_Pose_output'
path = 'TUM_testing_KITTI_Pose_output'
files = os.listdir(path)
files = natsort.natsorted(files)            #已经顺序排列可以不要

x = []
y = []
z = []

for file in files:
    with open(path+'/'+file) as f:
        lines = f.readlines()
        lines = lines[1]
        # print('lines[1]',type(lines))
        tx = lines.split(' ')[1]
        ty = lines.split(' ')[2]
        tz = lines.split(' ')[3]
        qx = lines.split(' ')[4]
        qy = lines.split(' ')[5]
        qz = lines.split(' ')[6]
        qw = lines.split(' ')[7]
        quat = [qw,qx,qy,qz]
        quat = map(float,quat)
        M = quat2mat(quat)                  #四元数到3*3的矩阵
        t = [tx,ty,tz]
        t = map(float,t)
        t = np.array(t).reshape(3,1)
        #
        if (file.split('.')[0] == '000000'):

            t0_x = [0.000000]
            t0_y = [0.000000]
            t0_z = [-0.000000]
            t0 = np.array([t0_x,t0_y,t0_z]).reshape(3,1)
            T = np.add(np.dot(M,t0),t)
        else:
            T = np.add(np.dot(M,T),t)
        # print('T',T)    #shape(3,1)


    x.append(T[0,0])
    y.append(T[1,0])
    z.append(T[2,0])

figure = plt.figure()

ax = figure.add_subplot(111,projection ='3d')
ax.plot(x, y, z)

#ax = figure.add_subplot(111)
#ax.plot(x, z)

plt.xlabel('x')
plt.ylabel('y')
plt.axis("equal")
plt.show()






