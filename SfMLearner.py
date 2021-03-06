#!/usr/bin/python27
#coding:utf-8
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *
import PIL.Image as pil
from Getdepth import *
img_height =128
img_width =416

class SfMLearner(object):
    def __init__(self):
        pass   #pass　
    
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)                        # [-1,1)
            src_image_stack = self.preprocess_image(src_image_stack)            # [-1,1)
            # tgt_image = tf.Print(tgt_image,[tgt_image],message= 'tgt_image')  #(4,128,416,3)
            # src_image_stack = tf.Print(src_image_stack, [src_image_stack], message='src_image_stack')     #(4,128,416,6)
            # # print('intrinsics:',intrinsics)                                   #shape(4,4,3,3)
            # print('tgt_image.shape', tgt_image.shape)                           # shape(4,128,416,3)
            # print('src_image_stack', src_image_stack.shape)                     # shape(4,128,416,6)

        with tf.name_scope("depth_prediction"):

            # [ disp:shape=[4,128,416,1]
            #Fix the groundtruth depth

            pred_disp = Getdepth()
            pred_disp = pred_disp.get_depth_graph()
            print('pred_disp:',pred_disp)


            pred_depth = [1./pred_disp]   # inverse depth
            # pred_depth = tf.Print(pred_depth,[pred_depth],message='pred_depth')
            print('pred_depth:',pred_depth)

        with tf.name_scope("pose_and_explainability_prediction"):
            pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                pose_exp_net(tgt_image,
                             src_image_stack, 
                             do_exp=(opt.explain_reg_weight > 0),
                             is_training=True)
            print('pred_poses.shape',pred_poses.shape)  #(4,2,6)
            # pred_poses = tf.Print(pred_poses,[pred_poses],message='pred_poses')
            # print('pred_exp_logits:',pred_exp_logits)
#loss function
        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            #todo:only one scale
            s = 0
            if opt.explain_reg_weight > 0:
                # Construct a reference explainability mask (i.e. all
                # pixels are explainable)
                ref_exp_mask = self.get_reference_explain_mask(s)
                # print('ref_exp_mask:',ref_exp_mask)
            # Scale the source and target images for computing loss at the
            # according scale.
            curr_tgt_image = tf.image.resize_area(tgt_image,
                [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])   #1个scale
            # print('curr_tgt_image:',curr_tgt_image)                     # (4, 128, 416, 3)
            curr_src_image_stack = tf.image.resize_area(src_image_stack,
                [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
            # print('curr_src_image_stack:', curr_src_image_stack)         # (4, 128, 416, 6)
            for i in range(opt.num_source):                     #num_source=2
                # Inverse warp the source image to the target image frame
                curr_proj_image = projective_inverse_warp(
                    curr_src_image_stack[:,:,:,3*i:3*(i+1)],                #shape(4,128,416,3)
                    tf.squeeze(pred_depth[s], axis=3),                      #shape(4,128,416)
                    pred_poses[:,i,:],                                      #pred_poses.shape(4,6)
                    intrinsics[:,s,:,:])                                    #shape(4,4,3,3)
                curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)  #投影光度误差
                # print('curr_proj_error.shape:',curr_proj_error.shape)
                # Cross-entropy loss as regularization for the
                # explainability prediction
                if opt.explain_reg_weight > 0:

                    curr_exp_logits = tf.slice(pred_exp_logits[s],
                                               [0, 0, 0, i*2],
                                               [-1, -1, -1, 2])
                    with tf.Session() as sess:
                        test_curr_exp_logits=sess.run(curr_exp_logits)
                        # print(test_curr_exp_logits)

                    exp_loss += opt.explain_reg_weight * \
                        self.compute_exp_reg_loss(curr_exp_logits,
                                                  ref_exp_mask)
                    curr_exp = tf.nn.softmax(curr_exp_logits)
                # Photo-consistency loss weighted by explainability
                if opt.explain_reg_weight > 0:
                    pixel_loss += tf.reduce_mean(curr_proj_error * \
                        tf.expand_dims(curr_exp[:,:,:,1], -1))
                else:
                    pixel_loss += tf.reduce_mean(curr_proj_error)
                # Prepare images for tensorboard summaries
                if i == 0:
                    proj_image_stack = curr_proj_image
                    # print('proj_image_stack.shape:',proj_image_stack.shape)
                    proj_error_stack = curr_proj_error
                    # print('proj_error_stack.shape:', proj_error_stack.shape)
                    if opt.explain_reg_weight > 0:
                        exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                else:
                    proj_image_stack = tf.concat([proj_image_stack,
                                                  curr_proj_image], axis=3)
                    # print('proj_image_stack.shape:',proj_image_stack.shape)
                    proj_error_stack = tf.concat([proj_error_stack,
                                                  curr_proj_error], axis=3)
                    # print('proj_error_stack.shape:', proj_error_stack.shape)
                    if opt.explain_reg_weight > 0:
                        exp_mask_stack = tf.concat([exp_mask_stack,
                            tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            total_loss = pixel_loss + exp_loss

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (opt.batch_size, 
                                int(opt.img_height/(2**downscaling)), 
                                int(opt.img_width/(2**downscaling)), 
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)


    def train(self, opt):
        opt.num_source = opt.seq_length - 1    #seq_length=3 ,source_view=2 and target_view=1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op
                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')                                           #shape(1,128,416*3,3)
        input_mc = self.preprocess_image(input_uint8)                   #shape(1,128,416*3,3)
        # print('input_mc.shape:',input_mc.shape)
        # input_mc = tf.Print(input_mc,[input_mc.shape],message='input_mc')
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)

        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            # print('pred_poses:',pred_poses)
            # tf.Print(pred_poses,[pred_poses.shape],message='pred_poses')         #shape(1,2,6)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        # print('image uint8: ', image)
        # image = tf.Print(image, [image], "image uint8: ")
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  #[0,1)的float
        # image = tf.Print(image, [image], "image float32: ")
        return image * 2. -1.  #(-1,1)

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.   #float32 to uint8
        return tf.image.convert_image_dtype(image, dtype=tf.uint8) #uint8

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}

        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

