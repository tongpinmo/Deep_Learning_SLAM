import gslam as gs
import numpy as np
import os
from PIL import Image
from test_pose_module2 import *
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np


class DeepGSLAM(gs.SLAM):
    def type(self):
        return "DeepSLAM";
    def valid(self):
        return True;
    def isDrawable(self):
        return False;
    def setSvar(self,var):
        self.config=var;
    def setCallback(self,cbk):
        self.callback=cbk;
    def track(self,fr):
        image=fr.getImage(0)
        print("Processing frame ",fr.id(),"time:",fr.timestamp())

        # format of numpy array
        image_np = np.zeros((image.height(), image.width(), image.channels()), np.uint8)

        for i in range(image.height()):
            for j in range(image.width()):
               id_b = i*image.width()*image.channels() + j*image.channels()
               image_np[i,j,0] = image.data(id_b+2)
               image_np[i,j,1] = image.data(id_b+1)
               image_np[i,j,2] = image.data(id_b)


        #fr.setImage(0,depth)
        #fr.setPose(gs.SE3())
        self.callback.handle(fr)
        return image_np

class GObjectHandle(gs.GObjectHandle):
    def handle(self,obj):
        print("Pose:",obj.getPose())


slam = DeepGSLAM()
dataset = gs.Dataset()
dataset.open("/mnt/PI_Lab/users/zhaoyong/Dataset/TUM/RGBD/rgbd_dataset_freiburg3_sitting_xyz/.tumrgbd")

callback = GObjectHandle()
slam.setCallback(callback)
fr = dataset.grabFrame()

sequence_len = 3
index_img = 0
seq_img = [0, 0, 0]


# 3 frames as input
#
# while(fr):
#     seq_img[index_img] = slam.track(fr)
#     # print('index_img:',index_img)
#     # print('seq_img[index_img]:',seq_img[index_img])
#     fr = dataset.grabFrame()
#     # print('fr.id():', fr.id())
#     index_img = index_img + 1
#     if index_img == sequence_len:
#         index_img = 0
#         # print('seq_img[0]',seq_img[0])
#         pose = get_pose(seq_img[0],seq_img[1],seq_img[2])



#continuous frames as input

indicate = 0
index = 0
while (fr):
    seq_img[index_img] = slam.track(fr)
    print('index_img:',index_img)
    print('seq_img[index_img]:',seq_img[index_img])
    fr = dataset.grabFrame()
    print('fr:',fr)
    index_img = index_img + 1
    index = index + 1
    if index >= sequence_len:
        index_img = 0
        if indicate > 0:
            seq_img[0], seq_img[1], seq_img[2] = seq_img[1], seq_img[2], seq_img[0]

        # call the pose function
        pose = get_pose(seq_img[0], seq_img[1], seq_img[2])
        indicate = 1







