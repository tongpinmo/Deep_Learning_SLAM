# !/usr/bin/python27
# coding:utf-8
from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from Getdepth import Getdepth

import os


def get():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    sfm = Getdepth()
    sfm.get_depth_graph()


if __name__ == '__main__':
    get()