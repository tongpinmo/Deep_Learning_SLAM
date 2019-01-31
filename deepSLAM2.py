import numpy as np
import os
from PIL import Image
from test_pose_module2 import *
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np

from gslam import *
import time


def showStringMsg(obj):
    print("object is ", obj)


def showStatus(status):
    print("Status:", status)


class DeepSLAM:
    def init(self, config):
        self.messenger = Messenger()
        self.messenger.subscribe(MapFrame, "images", 0, self.handleFrame)
        self.pubCurframe = self.messenger.advertise(MapFrame, "deep/curframe", 1, False)
        self.pubImage = self.messenger.advertise(GImage, "deep/curImage", 1, False)
        self.pubMap = self.messenger.advertise(Map, "deep/map", 1, False)
        self.pubGround = self.messenger.advertise(Map, "ground/map", 1, False)
        self.map = HashMap()
        self.ground = HashMap()
        self.imgSeq = [0,0,0]
        self.times = [0,0,0]
        self.lastPose = SE3()
        self.index = 0
        self.index_img = 0
        self.indicate = 0

        return self.messenger

    def SE3Pose(self,p):
        return SE3(SO3(p[4],p[5],p[6],p[7]),Point3d(p[1],p[2],p[3]))

    def handleFrame(self, fr):
        ground=MapFrame(fr.id(), fr.timestamp())
        ground.setPose(fr.getPose())
        self.ground.insertMapFrame(ground)

        image = fr.getImage(0)
        print("Processing frame ", fr.id(), "time:", fr.timestamp())

        imgnp = np.array(image,copy = False)
        print('imgnp.shape:',imgnp.shape)      #shape(480,640,3)

        self.imgSeq[self.index_img] = imgnp
        self.times[self.index_img] = fr.timestamp()

        # print('self.imgSeq:',self.imgSeq[self.index_img])
        print('self.index_img:',self.index_img)
        print('self.index:',self.index)


        self.index_img = self.index_img + 1
        self.index = self.index + 1


        if (self.index >= 3):
            self.index_img = 0

            if self.indicate > 0:
                self.imgSeq[0], self.imgSeq[1], self.imgSeq[2] = self.imgSeq[1],self.imgSeq[2],self.imgSeq[0]
                # print('len of self.imgSeq2:', len(self.imgSeq))
                self.times[0],self.times[1],self.times[2]  = self.times[1],self.times[2],self.times[0]
                print('self.times:', self.times)

            pose=get_pose(self.imgSeq, self.times, fr.id())
            print('pose:', pose)
            self.indicate = 1
            fr.setPose(self.SE3Pose(pose))


        self.map.insertMapFrame(fr)
        self.pubCurframe.publish(fr)
        self.pubImage.publish(image)
        self.pubMap.publish(self.map)
        self.pubGround.publish(self.ground)


msg = Messenger.singleton()
print(msg)
svar = Svar.singleton()

# svar.parseLine("Dataset=/mnt/PI_Lab/users/zhaoyong/Dataset/TUM/RGBD/rgbd_dataset_freiburg1_desk2/.tumrgbd")
svar.parseLine("Dataset=/mnt/PI_Lab/users/zhaoyong/Dataset/TUM/RGBD/rgbd_dataset_freiburg3_nostructure_texture_near_withloop/.tumrgbd")
# svar.parseLine("Dataset=/mnt/PI_Lab/users/zhaoyong/Dataset/TUM/RGBD/rgbd_dataset_freiburg3_long_office_household/.tumrgbd")

slam = DeepSLAM()
msg.accept(slam.init(svar))
print(svar)
print(msg)

qviz = Application.create("qviz")
print(type(qviz))

qviz.init(svar).accept(msg)

while not svar.getInt("ShouldStop"):
    time.sleep(0.1)




