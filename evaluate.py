# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:24:33 2018

@author: djj
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
import cv2
 
from test_image_reader import *
from net import *
 
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--x_test_data_path", default='/underwater/', help="path of x test datas.") #x域的测试图片路径
parser.add_argument("--y_test_data_path", default='/underwater/', help="path of y test datas.") #y域的测试图片路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--snapshots", default='./snapshots/',help="Path of Snapshots") #读取训练好的模型参数的路径
parser.add_argument("--out_dir_x", default='./test_output_x/',help="Output Folder") #保存x域的输入图片与生成的y域图片的路径
parser.add_argument("--out_dir_y", default='./test_output_y/',help="Output Folder") #保存y域的输入图片与生成的x域图片的路径
 
args = parser.parse_args()
 
def make_test_data_list(x_data_path, y_data_path): #make_test_data_list函数得到测试中的x域和y域的图像路径名称列表
    x_input_images = glob.glob(os.path.join(x_data_path, "*")) #读取全部的x域图像路径名称列表
    y_input_images = glob.glob(os.path.join(y_data_path, "*")) #读取全部的y域图像路径名称列表
    return x_input_images, y_input_images
 
def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) #bgr
 
def get_write_picture(x_image, y_image, fake_y, fake_x): #get_write_picture函数得到网络测试结果
    x_image = cv_inv_proc(x_image) #还原x域的图像
    y_image = cv_inv_proc(y_image) #还原y域的图像
    fake_y = cv_inv_proc(fake_y[0]) #还原生成的y域的图像
    fake_x = cv_inv_proc(fake_x[0]) #还原生成的x域的图像
    x_output = np.concatenate((x_image, fake_y), axis=1) #得到x域的输入图像以及对应的生成的y域图像
    y_output = np.concatenate((y_image, fake_x), axis=1) #得到y域的输入图像以及对应的生成的x域图像
    return x_output, y_output
 
def main():
    if not os.path.exists(args.out_dir_x): #如果保存x域测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir_x)
    if not os.path.exists(args.out_dir_y): #如果保存y域测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir_y)
 
    x_datalists, y_datalists = make_test_data_list(args.x_test_data_path, args.y_test_data_path) #得到待测试的x域和y域图像路径名称列表
    test_x_image = tf.placeholder(tf.float32,shape=[1, 256, 256, 3], name = 'test_x_image') #输入的x域图像
    test_y_image = tf.placeholder(tf.float32,shape=[1, 256, 256, 3], name = 'test_y_image') #输入的y域图像
 
    fake_y = generator(image=test_x_image, reuse=False, name='generator_x2y') #得到生成的y域图像
    fake_x = generator(image=test_y_image, reuse=False, name='generator_y2x') #得到生成的x域图像
 
    restore_var = [v for v in tf.global_variables() if 'generator' in v.name] #需要载入的已训练的模型参数
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #建立会话层
    
    saver = tf.train.Saver(var_list=restore_var,max_to_keep=1) #导入模型参数时使用
    checkpoint = tf.train.latest_checkpoint(args.snapshots) #读取模型参数
    saver.restore(sess, checkpoint) #导入模型参数
 
    total_step = len(x_datalists) if len(x_datalists) > len(y_datalists) else len(y_datalists) #测试的总步数
    for step in range(total_step):
        test_ximage_name, test_ximage = TestImageReader(x_datalists, step, args.image_size) #得到x域的输入及名称
        test_yimage_name, test_yimage = TestImageReader(y_datalists, step, args.image_size) #得到y域的输入及名称
        batch_x_image = np.expand_dims(np.array(test_ximage).astype(np.float32), axis = 0) #填充维度
        batch_y_image = np.expand_dims(np.array(test_yimage).astype(np.float32), axis = 0) #填充维度
        feed_dict = { test_x_image : batch_x_image, test_y_image : batch_y_image} #建立feed_dict
        fake_y_value, fake_x_value = sess.run([fake_y, fake_x], feed_dict=feed_dict) #得到生成的y域图像与x域图像
        x_write_image, y_write_image = get_write_picture(test_ximage, test_yimage, fake_y_value, fake_x_value) #得到最终的图片结果
        x_write_image_name = args.out_dir_x + "/"+ test_ximage_name + ".png" #待保存的x域图像与其对应的y域生成结果名字
        y_write_image_name = args.out_dir_y + "/"+ test_yimage_name + ".png" #待保存的y域图像与其对应的x域生成结果名字
        cv2.imwrite(x_write_image_name, x_write_image) #保存图像
        cv2.imwrite(y_write_image_name, y_write_image) #保存图像
        print('step {:d}'.format(step))
 
if __name__ == '__main__':
    main()
