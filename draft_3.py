#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:22:00 2018

@author: usrp1
"""
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import random
import numpy as np
import argparse
import tensorflow as tf
import cv2

def hue_similar(img1,img2):
    img1_hsv = tf.image.rgb_to_hsv(img1);
    img1_h=tf.floor(tf.multiply(tf.add(img1_hsv[:,:,0],0),255))
    img1_s=tf.floor(tf.multiply(tf.add(img1_hsv[:,:,1],0),255))
    img1_v=tf.floor(tf.multiply(tf.add(img1_hsv[:,:,2],0),255))
    
    hist_1_h = tf.histogram_fixed_width(values=img1_h,nbins=256,value_range=[0.0,255.0],name='hist_1_h',dtype=tf.float32)
    hist_1_h_norm = tf.multiply(hist_1_h,1.0/65536,name='hist_1_h_norm')
    
    hist_1_s = tf.histogram_fixed_width(values=img1_s,nbins=256,value_range=[0.0,255.0],name='hist_1_s',dtype=tf.float32)
    hist_1_s_norm = tf.multiply(hist_1_s,1.0/65536,name='hist_1_s_norm')
    
    hist_1_v = tf.histogram_fixed_width(values=img1_v,nbins=256,value_range=[0.0,255.0],name='hist_1_v',dtype=tf.float32)
    hist_1_v_norm = tf.multiply(hist_1_v,1.0/65536,name='hist_1_v_norm')

    img2_hsv = tf.image.rgb_to_hsv(img2);
    img2_h=tf.floor(tf.multiply(tf.add(img2_hsv[:,:,0],0),255))
    img2_s=tf.floor(tf.multiply(tf.add(img2_hsv[:,:,1],0),255))
    img2_v=tf.floor(tf.multiply(tf.add(img2_hsv[:,:,2],0),255))
    
    hist_2_h = tf.histogram_fixed_width(values=img2_h,nbins=256,value_range=[0.0,255.0],name='hist_2_h',dtype=tf.float32)
    hist_2_h_norm = tf.multiply(hist_2_h,1.0/65536,name='hist_2_h_norm')
    
    hist_2_s = tf.histogram_fixed_width(values=img2_s,nbins=256,value_range=[0.0,255.0],name='hist_2_s',dtype=tf.float32)
    hist_2_s_norm = tf.multiply(hist_2_s,1.0/65536,name='hist_2_s_norm')
    
    hist_2_v = tf.histogram_fixed_width(values=img2_v,nbins=256,value_range=[0.0,255.0],name='hist_2_v',dtype=tf.float32)
    hist_2_v_norm = tf.multiply(hist_2_v,1.0/65536,name='hist_2_v_norm')
    
    index_h =1- tf.reduce_sum(tf.minimum(hist_1_h_norm,hist_2_h_norm))
    index_s =1- tf.reduce_sum(tf.minimum(hist_1_s_norm,hist_2_s_norm))
    index_v =1- tf.reduce_sum(tf.minimum(hist_1_v_norm,hist_2_v_norm))

    return (0.5*index_h+0.25*index_s+0.25*index_v)


x=tf.placeholder(tf.float32,shape=[1],name='x')
r = tf.cond(tf.less(x[0], tf.constant(0.5)), lambda:tf.multiply(x[0],2.0), lambda:tf.multiply(x[0],3.0))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #设定显存不超量使用
sess = tf.Session(config=config) #新建会话层
init = tf.global_variables_initializer() #参数初始化器
 
sess.run(init) #初始化所有可训练参数
#b = random.sample(range(15),13)
#djj = hue_similar(np.uint8(y_image_RGB),np.uint8(x_image_RGB))
#y_image = cv2.imread('//home//usrp1//djj_cycle_GAN//img//apple2orange//testA//2.jpg')