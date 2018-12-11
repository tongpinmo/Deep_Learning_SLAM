import numpy as np
import os
from PIL import Image
from test_pose_module1 import *
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
        self.map = HashMap()
        self.imgSeq = []
        self.times = []
        self.lastPose = SE3()
        return self.messenger

    def SE3Pose(self,p):
        return SE3(p[1],p[2],p[3],p[4],p[5],p[6],p[7])

    def handleFrame(self, fr):
        image = fr.getImage(0)
        cam = fr.getCamera(0)
        print("Processing frame ", fr.id(), "time:", fr.timestamp())

        imgnp = np.array(image,copy = False)
        # print('imgnp.shape:',imgnp.shape)                           #shape(480,640,3)

        indicate = 0
        if (len(self.imgSeq) >= 3):
            self.imgSeq.append(imgnp)
            self.times.append(fr.timestamp())

            if indicate > 0:

                self.imgSeq = [self.imgSeq[1],self.imgSeq[2],imgnp]
                # print('len of self.imgSeq2:', len(self.imgSeq))
                self.times  = [self.times[1],self.times[2],fr.timestamp()]
                # print('self.times:', self.times)


            pose=get_pose(self.imgSeq, self.times, fr.id())
            print('pose:', pose)
            indicate = 1
            fr.setPose(self.SE3Pose(pose[0]))

        self.map.insertMapFrame(fr)
        self.pubCurframe.publish(fr)
        self.pubImage.publish(image)
        self.pubMap.publish(self.map)

msg = Messenger.singleton()
svar = Svar.singleton()

svar.parseLine("Dataset=/mnt/PI_Lab/users/zhaoyong/Dataset/TUM/RGBD/rgbd_dataset_freiburg1_360/.tumrgbd")

slam = DeepSLAM()
msg.accept(slam.init(svar))

qviz = Application.create("qviz")
qviz.init(svar).accept(msg)

while not svar.getInt("ShouldStop"):
    time.sleep(0.1)




