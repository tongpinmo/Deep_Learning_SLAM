# !/usr/bin/python27
# coding:utf-8

from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import *
import matplotlib.pyplot as plt
import functools


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size  of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length",3, "Sequence length for each example")
flags.DEFINE_string("ckpt_file", 'checkpoints_NYU/model-19950', "checkpoint file")
flags.DEFINE_string("output_dir", 'deepSLAMpose_output/', "Output directory")
FLAGS = flags.FLAGS

#load the checkpoint file
sfm = SfMLearner()  # __init__
sfm.setup_inference(FLAGS.img_height,
                    FLAGS.img_width,
                    'pose',
                    FLAGS.seq_length)

if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
saver = tf.train.Saver([var for var in tf.trainable_variables()])

#FIXME:此种方式不会自动close session
sess = tf.Session()
saver.restore(sess, FLAGS.ckpt_file)

##这种方式会自动关闭session
# with tf.Session() as sess:
#     saver.restore(sess, FLAGS.ckpt_file)

def get_pose(img,times,index):

    max_src_offset = (FLAGS.seq_length - 1)//2   #1

    tgt_idx = 1
    # TODO: currently assuming batch_size = 1
    image_seq = load_image_sequence(img,
                                    tgt_idx,
                                    FLAGS.seq_length,
                                    FLAGS.img_height,
                                    FLAGS.img_width)
    # print('image_seq.shape',image_seq.shape)                      #(128, 1248, 3)
    # 传入data,feed_dict={}
    pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')       #an dictionary
    # print('pred_poses.array:',pred['pose'])
    # print('pred_poses.array:', pred['pose'].shape)                  #shape(1,2,6)
    pred_poses = pred['pose'][0]   # dictionary to ndarray
    # print('pred_poses.shape:',pred_poses.shape)                   #shape(2,6)
    # Insert the target pose [0, 0, 0, 0, 0, 0]
    pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)  # the target image is the reference
    # print('pred_poses',pred_poses)
    # print('pred_poses[0]:',pred_poses[0])
    poses = dump_pose_seq_TUM(pred_poses,times)
    pose_final = gen_TUM_format_pose(poses,index)

    return pose_final


def gen_TUM_format_pose(poses, index):
    time = poses[1][0]
    print('time:',time)
    tx = poses[1][1]
    ty = poses[1][2]
    tz = poses[1][3]
    qx = poses[1][4]
    qy = poses[1][5]
    qz = poses[1][6]
    qw = poses[1][7]

    quat = np.array([qw, qx, qy, qz])
    R = quat2mat(quat)  # shape(3,3)

    t = np.array([tx, ty, tz])
    t = t.reshape(3, 1)

    Tmat = TUM_vec_to_Tmat(R, t)
    # print('Tmat:',Tmat)
    
    if (index == 3):
        print('poses[0]:',poses[0])
        print('poses[1]:',poses[1])
        pose_final = poses[0]
        pose_final.append(poses[1])
        global this_pose
        this_pose = Tmat

        
    else:

        this_pose = np.dot(Tmat,this_pose)
        # print('this_pose:',this_pose)
        tx = this_pose[0, 3]
        ty = this_pose[1, 3]
        tz = this_pose[2, 3]
        rot = this_pose[:3, :3]
        qw, qx, qy, qz = rot2quat(rot)

        pose_final = [time,tx,ty,tz,qx,qy,qz,qw]

    return pose_final



#arrange three images into a sequence
def load_image_sequence(img,
                        tgt_idx,
                        seq_length,
                        img_height,
                        img_width):
    # print('img:',img)
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        tgt_idx = 1
        curr_idx = tgt_idx + o

        curr_img = img[curr_idx]

        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width)) #调整图像尺寸

        if o == -half_offset:
            image_seq = curr_img
            # print('image_seq.0',image_seq.shape)            # ('image_seq.0', (128, 416, 3))
        else:
            image_seq = np.hstack((image_seq, curr_img))
            # print('image_seq:',image_seq.shape)
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False


def dump_pose_seq_TUM(pred_poses,times):
    # First frame as the origin
    first_pose = pose_vec_to_mat(pred_poses[0])
    # print('first_pose:',first_pose)
    poses = []
    for p in range(len(times)):
        this_pose = pose_vec_to_mat(pred_poses[p])
        this_pose = np.dot(first_pose, np.linalg.inv(this_pose))                  #FIXME：change other frame into t-1 base
        tx = this_pose[0, 3]
        ty = this_pose[1, 3]
        tz = this_pose[2, 3]
        rot = this_pose[:3, :3]
        qw, qx, qy, qz = rot2quat(rot)
        # print('times,tx,ty,tz,qx,qy,qz,qw :',times[p],tx,ty,tz,qx,qy,qz,qw)
        re_pose = [times[p], tx, ty, tz, qx, qy, qz, qw]
        # print('re_pose:', re_pose)
        poses.append(re_pose)
    # print('poses:',poses)
    # print('pose[0]:',poses[0])
    # print('poses[1]:',poses[1])
    # print('pose[2]:',poses[2])
    return poses

#将Poses(tx,ty,tz,qx,qy,qz,qw)转化成4*4的变换矩阵T
def TUM_vec_to_Tmat(R,t):

    Tmat = np.concatenate((R,t), axis=1) # shape(3,4)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat

def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1) # shape(3,4)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat


def euler2mat(z=0, y=0, x=0, isRadian=True):
    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x

    Ms = []
    if z:
            cosz = math.cos(z)
            sinz = math.sin(z)
            Ms.append(np.array(
                            [[cosz, -sinz, 0],
                             [sinz, cosz, 0],
                             [0, 0, 1]]))
    if y:
            cosy = math.cos(y)
            siny = math.sin(y)
            Ms.append(np.array(
                            [[cosy, 0, siny],
                             [0, 1, 0],
                             [-siny, 0, cosy]]))
    if x:
            cosx = math.cos(x)
            sinx = math.sin(x)
            Ms.append(np.array(
                            [[1, 0, 0],
                             [0, cosx, -sinx],
                             [0, sinx, cosx]]))
    if Ms:
            return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def rot2quat(R):
    rz, ry, rx = mat2euler(R)
    qw, qx, qy, qz = euler2quat(rz, ry, rx)
    return qw, qx, qy, qz

def mat2euler(M, cy_thresh=None, seq='zyx'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if seq=='zyx':
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
    elif seq=='xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = math.atan2(r12, r13)
            else:
                y = -np.pi/2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x


def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
         Quaternion in w, x, y z (real, then vector) format
    Notes
    -----
    We can derive this formula in Sympy using:
    1. Formula giving quaternion corresponding to rotation of theta radians
         about arbitrary axis:
         http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
         theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
         http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
         formulae from 2.) to give formula for combined rotations.
    '''

    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])








