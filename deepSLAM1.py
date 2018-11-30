import gslam as gs
import numpy as np
import os
from PIL import Image
from test_pose_module1 import *
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
    def get_image(self,fr):
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
        # fr.setPose(gs.SE3(tx,ty,tz,qx,qy,qz,qw))
        self.callback.handle(fr)
        return image_np

class GObjectHandle(gs.GObjectHandle):
    def handle(self,obj):
        print("Pose:",obj.getPose())


slam = DeepGSLAM()
dataset = gs.Dataset()
dataset.open("/mnt/PI_Lab/users/zhaoyong/Dataset/TUM/RGBD/rgbd_dataset_freiburg1_360/.tumrgbd")

callback = GObjectHandle()
slam.setCallback(callback)
fr = dataset.grabFrame()

sequence_len = 3
index_img = 0
seq_img = [0, 0, 0]



#continuous frames as input

indicate = 0
index = 0
index_img = 0
times = [0,0,0]

while (fr):
    seq_img[index_img] = slam.get_image(fr)
    # print('index_img:', index_img)
    times[index_img] = fr.timestamp()
    # print('fr.timestamp:',times)
    index_img = index_img + 1
    # print('index_img:', index_img)
    index = index + 1
    # print('index:',index)
    if index >= sequence_len:
        index_img = 0
        if indicate > 0:
            # set the new frame in the seq_img[0],new timestamp in time[0]
            seq_img[0], seq_img[1], seq_img[2] = seq_img[1], seq_img[2], seq_img[0]
            times[0],times[1],times[2] = times[1],times[2],times[0]


        # call the pose function
        pose = get_pose(seq_img,index,times)
        print('times:',times)
        indicate = 1


    fr = dataset.grabFrame()







