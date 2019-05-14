# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:23:52 2018

@author: djj
"""

import os
import random
import numpy as np
import tensorflow as tf
import cv2
 
def TrainImageReader(x_file_list, y_file_list, step, size): #训练数据读取接口
    file_length = len(x_file_list) #获取图片列表总长度
    position = list(range(file_length))
    choice_position = random.sample(position,1)
    x_out_image = np.ones((1,size,size,3))
    y_out_image = np.ones((1,size,size,3))
    for i in range(1):
        line_idx = choice_position[i] % file_length #获取一张待读取图片的下标
        x_line_content = x_file_list[line_idx] #获取一张x域图片路径与名称
        y_line_content = y_file_list[line_idx] #获取一张y域图片路径与名称
        x_image = np.float32(cv2.imread(x_line_content,1)) #读取一张x域的图片
        y_image = np.float32(cv2.imread(y_line_content,1)) #读取一张y域的图片
        x_image_resize_t = cv2.resize(x_image, (size, size),interpolation=cv2.INTER_AREA) #改变读取的x域图片的大小
        x_image_resize = x_image_resize_t/127.5-1. #归一化x域的图片
        y_image_resize_t = cv2.resize(y_image, (size, size),interpolation=cv2.INTER_AREA) #改变读取的y域图片的大小
        y_image_resize = y_image_resize_t/127.5-1. #归一化y域的图片
        x_out_image[i,:,:,:]=x_image_resize
        y_out_image[i,:,:,:]=y_image_resize
    return x_out_image, y_out_image #返回读取并处理的一张x域图片和y域图片
