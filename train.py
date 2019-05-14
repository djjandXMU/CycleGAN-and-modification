# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:22:51 2018

@author: djj
"""

from __future__ import print_function
 
#from numba import jit
import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
import cv2
from skimage.measure import _structural_similarity as ssim
#import random
#import TrainImageReader
 
from train_image_reader import *
from net import *
 
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--snapshot_dir", default='./snapshots', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') #基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=4000, help='# of epoch') #训练的epoch数量
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=4000, help='# of epoch to decay lr') #训练中保持学习率不变的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=10000, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=500, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=100000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--x_train_data_path", default='//home/usrp1/djj_cycle_GAN/new_color_cnn_40/data_set_x/', help="path of x training datas.") #x域的训练图片路径
parser.add_argument("--y_train_data_path", default='//home/usrp1/djj_cycle_GAN/new_color_cnn_40/data_set_y/', help="path of y training datas.") #y域的训练图片路径
args = parser.parse_args()
 
#def save(saver, sess, logdir, step): #保存模型的save函数
#   model_name = 'model' #保存的模型名前缀
#   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
#   if not os.path.exists(logdir): #如果路径不存在即创建
#      os.makedirs(logdir)
#   saver.save(sess, checkpoint_path, global_step=step) #保存模型
#   print('The checkpoint has been created.')
 
def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像

def get_write_picture(x_image, y_image, fake_y, fake_x_, fake_x, fake_y_): #get_write_picture函数得到训练过程中的可视化结果
    x_image = cv_inv_proc(x_image) #还原x域的图像
    y_image = cv_inv_proc(y_image) #还原y域的图像
    x_image = x_image[0]
    y_image = y_image[0]
    fake_y = cv_inv_proc(fake_y[0]) #还原生成的y域的图像
    fake_x_ = cv_inv_proc(fake_x_[0]) #还原重建的x域的图像
    fake_x = cv_inv_proc(fake_x[0]) #还原生成的x域的图像
    fake_y_ = cv_inv_proc(fake_y_[0]) #还原重建的y域的图像
    row1 = np.concatenate((x_image, fake_y, fake_x_), axis=1) #得到训练中可视化结果的第一行
    row2 = np.concatenate((y_image, fake_x, fake_y_), axis=1) #得到训练中可视化结果的第二行
    output = np.concatenate((row1, row2), axis=0) #得到训练中可视化结果
    return output
 
def make_train_data_list(x_data_path, y_data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    x_input_images_raw = glob.glob(os.path.join(x_data_path, "*")) #读取全部的x域图像路径名称列表
    y_input_images_raw = glob.glob(os.path.join(y_data_path, "*")) #读取全部的y域图像路径名称列表
    x_input_images, y_input_images = add_train_list(x_input_images_raw, y_input_images_raw) #将x域图像数量与y域图像数量对齐
    return x_input_images, y_input_images
 
def add_train_list(x_input_images_raw, y_input_images_raw): #add_train_list函数将x域和y域的图像数量变成一致
    if len(x_input_images_raw) == len(y_input_images_raw): #如果x域和y域图像数量本来就一致，直接返回
        shuffle(x_input_images_raw)
        shuffle(y_input_images_raw)
        return x_input_images_raw,y_input_images_raw
    elif len(x_input_images_raw) > len(y_input_images_raw): #如果x域的训练图像数量大于y域的训练图像数量，则随机选择y域的图像补充y域
        mul_num = int(len(x_input_images_raw)/len(y_input_images_raw)) #计算两域图像数量相差的倍数
        y_append_num = len(x_input_images_raw) - len(y_input_images_raw)*mul_num #计算需要随机出的y域图像数量
        append_list = [random.randint(0,len(y_input_images_raw)-1) for i in range(y_append_num)] #得到需要补充的y域图像下标
        y_append_images = [] #初始化需要被补充的y域图像路径名称列表
        for a in append_list:
            y_append_images.append(y_input_images_raw[a])
        y_input_images = y_input_images_raw * mul_num + y_append_images #得到数量与x域一致的y域图像
        shuffle(x_input_images_raw) #随机打乱x域图像顺序
        shuffle(y_input_images) #随机打乱y域图像顺序
        return x_input_images_raw, y_input_images #返回数量一致的x域和y域图像路径名称列表
    else: #与elif中的逻辑一致，只是x与y互换，不再赘述
        mul_num = int(len(y_input_images_raw)/len(x_input_images_raw))
        x_append_num = len(y_input_images_raw) - len(x_input_images_raw)*mul_num
        append_list = [random.randint(0,len(x_input_images_raw)-1) for i in range(x_append_num)]
        x_append_images = []
        for a in append_list:
            x_append_images.append(x_input_images_raw[a])
        x_input_images = x_input_images_raw * mul_num + x_append_images
        shuffle(y_input_images_raw)
        shuffle(x_input_images)
        return x_input_images, y_input_images_raw
#@jit   
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))
#@jit
def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src-dst)**2)

def var(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
#@jit
def ssim_tf(image_1,image_2,c_1 = 0.01,c_2 = 0.03):
    image_1=(image_1+1)*127.5
    image_2=(image_2+1)*127.5
    image_1 =tf.reshape(image_1,[1,256,256,1])
    image_2 =tf.reshape(image_2,[1,256,256,1])
#    gauss_filter=tf.constant(np.random.normal(loc=0,scale=1,size=(5,5,1,1)),name = 'gauss_filter',dtype=tf.float32)
    gauss_filter = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1])/273.0
    gauss_filter.reshape((5,5,1,1))##我先用的5*5的滤波器###################################
    gauss_filter = tf.constant(value =gauss_filter ,shape=[5, 5, 1, 1],dtype=tf.float32,name='gauss_filter')
    #gauss_filter=np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273.0
    #gauss_filter = tf.constant(gauss_filter,name='gauss_filter',dtype=tf.float32)
    image_1_u = tf.nn.conv2d(image_1,gauss_filter, [1, 1, 1, 1], padding = 'SAME')
    image_1_u2 = tf.multiply(image_1_u,image_1_u)
    
    image_2_u = tf.nn.conv2d(image_2,gauss_filter, [1, 1, 1, 1], padding = 'SAME')
    image_2_u2 = tf.multiply(image_2_u,image_2_u)
    
    image_u1_u2 = tf.multiply(image_2_u,image_1_u)
    
    var_image_1 = tf.nn.conv2d(tf.multiply(image_1,image_1),gauss_filter, [1, 1, 1, 1], padding = 'SAME') -image_1_u2
    var_image_2 = tf.nn.conv2d(tf.multiply(image_2,image_2),gauss_filter, [1, 1, 1, 1], padding = 'SAME') -image_2_u2
    var_image_12 = tf.nn.conv2d(tf.multiply(image_1,image_2),gauss_filter, [1, 1, 1, 1], padding = 'SAME') -image_u1_u2
    #ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2)); 
    c_1 = (0.01*255)*(0.01*255)###
    c_2 = (0.03*255)*(0.03*255)###
    ssim_map = tf.multiply(tf.divide((2* image_u1_u2+c_1),(image_1_u2 +image_2_u2 + c_1 )),tf.divide((2*var_image_12+c_2),(var_image_1+var_image_2+c_2)))
    ssim_ch=1-tf.reduce_mean(ssim_map)
    return ssim_ch



def color_similar(img1,img2):
    img1 = tf.multiply(tf.add(img1,1.0),127.5)
    img2 = tf.multiply(tf.add(img2,1.0),127.5)
    
    img1_r=tf.floor(tf.multiply(tf.add(img1[0,:,:,0],0),255))
    img1_g=tf.floor(tf.multiply(tf.add(img1[0,:,:,1],0),255))
    img1_b=tf.floor(tf.multiply(tf.add(img1[0,:,:,2],0),255))
    
    img2_r=tf.floor(tf.multiply(tf.add(img2[0,:,:,0],0),255))
    img2_g=tf.floor(tf.multiply(tf.add(img2[0,:,:,1],0),255))
    img2_b=tf.floor(tf.multiply(tf.add(img2[0,:,:,2],0),255))

    hist_1_r = tf.histogram_fixed_width(values=img1_r,nbins=256,value_range=[0.0,255.0],name='hist_1_r',dtype=tf.float32)
    hist_1_g = tf.histogram_fixed_width(values=img1_g,nbins=256,value_range=[0.0,255.0],name='hist_1_g',dtype=tf.float32)
    hist_1_b = tf.histogram_fixed_width(values=img1_b,nbins=256,value_range=[0.0,255.0],name='hist_1_b',dtype=tf.float32)
    
    hist_2_r = tf.histogram_fixed_width(values=img2_r,nbins=256,value_range=[0.0,255.0],name='hist_2_r',dtype=tf.float32)
    hist_2_g = tf.histogram_fixed_width(values=img2_g,nbins=256,value_range=[0.0,255.0],name='hist_2_g',dtype=tf.float32)
    hist_2_b = tf.histogram_fixed_width(values=img2_b,nbins=256,value_range=[0.0,255.0],name='hist_2_b',dtype=tf.float32)

#    print(hist_1)
    hist_1_r_norm = tf.multiply(hist_1_r,1.0/65536,name='hist_1_r_norm')
    hist_1_g_norm = tf.multiply(hist_1_g,1.0/65536,name='hist_1_g_norm')
    hist_1_b_norm = tf.multiply(hist_1_b,1.0/65536,name='hist_1_b_norm')
    
    hist_2_r_norm = tf.multiply(hist_2_r,1.0/65536,name='hist_2_r_norm')
    hist_2_g_norm = tf.multiply(hist_2_g,1.0/65536,name='hist_2_g_norm')
    hist_2_b_norm = tf.multiply(hist_2_b,1.0/65536,name='hist_2_b_norm')
    


    index_r = tf.reduce_sum(tf.minimum(hist_1_r_norm,hist_2_r_norm))
    index_g = tf.reduce_sum(tf.minimum(hist_1_g_norm,hist_2_g_norm))
    index_b = tf.reduce_sum(tf.minimum(hist_1_b_norm,hist_2_b_norm))
    index = (index_r+index_g+index_b)/3.0
#    k=1
#    for i in range(256):
#        for j in range(256):
#            tf.add(hist_1[0,img1[i,j]],1)
#            tf.add(hist_2[0,img2[i,j]],1)
#            print(k)
#            k=k+1
#
    return 1-index


################################################################################################
################################################################################################
#@jit
def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=5, sigma=1.5):
    gauss_filter = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1])/273.0
    gauss_filter.reshape((5,5,1,1))##我先用的5*5的滤波器###################################
    window = tf.constant(value =gauss_filter ,shape=[5, 5, 1, 1],dtype=tf.float32,name='window')
#    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 255  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)) ###### SSIM_map,C(X,Y)
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

#@jit
def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    img1 =tf.reshape(img1,[1,256,256,1])
    img2 =tf.reshape(img2,[1,256,256,1])
    img1=(img1+1)*127.5
    img2=(img2+1)*127.5
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

#    c=mcs[0:level-1]**weight[0:level-1]
#    print(c)
#    b=tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])
    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[0:level]**weight[0:level]))

    if mean_metric:
        value = 1-tf.reduce_mean(value)
    return value
################################################################################################
################################################################################################
#@jit
def hue_similar(img1,img2):
    img1 = tf.multiply(tf.add(img1,1.0),127.5)
    img2 = tf.multiply(tf.add(img2,1.0),127.5)
    
    img1_hsv = tf.image.rgb_to_hsv(img1);
    img1_h=tf.floor(tf.multiply(tf.add(img1_hsv[:,:,0],0),255))
    img1_s=tf.floor(tf.multiply(tf.add(img1_hsv[:,:,1],0),255.0))
    img1_v=tf.floor(tf.multiply(tf.add(img1_hsv[:,:,2],0),255.0))
    
    hist_1_h = tf.histogram_fixed_width(values=img1_h,nbins=256,value_range=[0.0,255.0],name='hist_1_h',dtype=tf.int32)
    hist_1_h_norm = tf.multiply(tf.cast(hist_1_h,dtype=tf.float32),1.0/65536,name='hist_1_h_norm')
    
    hist_1_s = tf.histogram_fixed_width(values=img1_s,nbins=256,value_range=[0.0,255.0],name='hist_1_s',dtype=tf.int32)
    hist_1_s_norm = tf.multiply(tf.cast(hist_1_s,dtype=tf.float32),1.0/65536,name='hist_1_s_norm')
    
    hist_1_v = tf.histogram_fixed_width(values=img1_v,nbins=256,value_range=[0.0,255.0],name='hist_1_v',dtype=tf.int32)
    hist_1_v_norm = tf.multiply(tf.cast(hist_1_v,dtype=tf.float32),1.0/65536,name='hist_1_v_norm')

    img2_hsv = tf.image.rgb_to_hsv(img2);
    img2_h=tf.floor(tf.multiply(tf.add(img2_hsv[:,:,0],0),255))
    img2_s=tf.floor(tf.multiply(tf.add(img2_hsv[:,:,1],0),255))
    img2_v=tf.floor(tf.multiply(tf.add(img2_hsv[:,:,2],0),255))
    
    hist_2_h = tf.histogram_fixed_width(values=img2_h,nbins=256,value_range=[0.0,255.0],name='hist_2_h',dtype=tf.int32)
    hist_2_h_norm = tf.multiply(tf.cast(hist_2_h,dtype=tf.float32),1.0/65536,name='hist_2_h_norm')
    
    hist_2_s = tf.histogram_fixed_width(values=img2_s,nbins=256,value_range=[0.0,255.0],name='hist_2_s',dtype=tf.int32)
    hist_2_s_norm = tf.multiply(tf.cast(hist_2_s,dtype=tf.float32),1.0/65536,name='hist_2_s_norm')
    
    hist_2_v = tf.histogram_fixed_width(values=img2_v,nbins=256,value_range=[0.0,255.0],name='hist_2_v',dtype=tf.int32)
    hist_2_v_norm = tf.multiply(tf.cast(hist_2_v,dtype=tf.float32),1.0/65536,name='hist_2_v_norm')
    
    index_h =1- tf.reduce_sum(tf.minimum(hist_1_h_norm,hist_2_h_norm))
    index_s =1- tf.reduce_sum(tf.minimum(hist_1_s_norm,hist_2_s_norm))
    index_v =1- tf.reduce_sum(tf.minimum(hist_1_v_norm,hist_2_v_norm))

    return (0.3*index_h+0.3*index_s+0.3*index_v)
#@jit
def color_index_adjust(index):###系数范围在0-1，放大10倍进行加权
#    return tf.cond(index < tf.constant(0.25),lambda:0.0,lambda:tf.div(10.0,tf.add(tf.exp(tf.add(tf.multiply(index,-10.0),5.0)),1.5))+0.3)
#    return loss

#    if (tf.less_equal(index,0.25)):
#        return 0
#    else:
#    return tf.div(10.0,tf.add(tf.exp(tf.add(tf.multiply(index,-10.0),5.0)),1.33))
    return tf.div(10.0,tf.add(tf.exp(tf.add(tf.multiply(index,-10.0),5.0)),1.0))+1
#@jit
def ssim_index_adjust(index):
    
    return tf.div(10.0,tf.add(tf.exp(tf.add(tf.multiply(index,-10.0),5.0)),1.5))
#    return tf.multiply(index,10.0)
def white_img(x):
    x = (x+1)*127.5
    x_correc=np.zeros((1,256,256,3))
    for i in range(1):
        r_mean = np.mean(x[i,:,:,0])
        g_mean = np.mean(x[i,:,:,1])
        b_mean = np.mean(x[i,:,:,2])
        
        r_mean_ratio = 1/(r_mean/127.5)
        g_mean_ratio = 1/(g_mean/127.5)
        b_mean_ratio = 1/(b_mean/127.5)
        
        x_correc[i,:,:,0] = x[i,:,:,0]*r_mean_ratio
        x_correc[i,:,:,1] = x[i,:,:,1]*g_mean_ratio
        x_correc[i,:,:,2] = x[i,:,:,2]*b_mean_ratio
        
    return ((x_correc/127.5)-1)



def main():
#    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
#        os.makedirs(args.snapshot_dir)
#    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
#        os.makedirs(args.out_dir)
    x_datalists, y_datalists = make_train_data_list(args.x_train_data_path, args.y_train_data_path) #得到数量相同的x域和y域图像路径名称列表
    tf.set_random_seed(args.random_seed) #初始一下随机数
    x_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='x_img') #输入的x域图像
    y_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='y_img') #输入的y域图像
    
    
    
#    buffer_Dx = tf.Variable(tf.truncated_normal([15,70,70,3]),name = 'buffuer_Dx')
#    buffer_Dy = tf.Variable(tf.truncated_normal([15,70,70,3]),name = 'buffuer_Dy')
 ################image sizes are n*row*col*channel
    fake_y = generator(image=x_img, reuse=False, name='generator_x2y') #生成的y域图像
    fake_x_ = generator(image=fake_y, reuse=False, name='generator_y2x') #重建的x域图像
    fake_x = generator(image=y_img, reuse=True, name='generator_y2x') #生成的x域图像
    fake_y_ = generator(image=fake_x, reuse=True, name='generator_x2y') #重建的y域图像
    
#    input_fake_y = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_y')
#    input_fake_x = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_x')
#    input_fake_y_ = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_y_')
#    input_fake_x_ = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_x_')
#    input_x_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_x_img')
#    input_y_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_y_img')
    
#    color_index_G = hue_similar(fake_y[0,:,:,:],y_img[0,:,:,:])
#    color_index_cy_G = hue_similar(fake_y_[0,:,:,:],y_img[0,:,:,:])
#    color_loss_G =color_index_adjust(tf.abs(0.9*color_index_G+0.1*color_index_cy_G-0.3))
    
#    color_index_F = hue_similar(fake_x[0,:,:,:],x_img[0,:,:,:])
#    color_index_cy_F = hue_similar(fake_x_[0,:,:,:],x_img[0,:,:,:])
#    color_loss_F =color_index_adjust(0.9*color_index_F+0.1*color_index_cy_F)
#    color_index_adjust
  ####tf_ms_ssim(img1, img2, mean_metric=True, level=5)  
#    ssim_loss_1_x = ssim_tf(fake_y[0,:,:,0],x_img[0,:,:,0])
#    ssim_loss_2_x = ssim_tf(fake_y[0,:,:,1],x_img[0,:,:,1])
#    ssim_loss_3_x = ssim_tf(fake_y[0,:,:,2],x_img[0,:,:,2])
#  tf.add(tf.add(tf.div(fake_y[0,:,:,0],3),tf.div(fake_y[0,:,:,1],3)),tf.div(fake_y[0,:,:,2],3))
    fake_y_gray = (fake_y[0,:,:,0]+fake_y[0,:,:,1]+fake_y[0,:,:,2])/3 
    x_img_gray  = (x_img[0,:,:,0]+x_img[0,:,:,1]+x_img[0,:,:,2])/3
    ssim_loss_x = ssim_tf(fake_y_gray,x_img_gray)
#    ssim_loss_x =ssim_index_adjust( ssim_loss_x )
#    
#    ssim_loss_1_y = ssim_tf(y_img[0,:,:,0],fake_x[0,:,:,0])
#    ssim_loss_2_y = ssim_tf(y_img[0,:,:,1],fake_x[0,:,:,1])
#    ssim_loss_3_y = ssim_tf(y_img[0,:,:,2],fake_x[0,:,:,2])
    fake_x_gray = (fake_x[0,:,:,0]+fake_x[0,:,:,1]+fake_x[0,:,:,2])/3 
    y_img_gray  = (y_img[0,:,:,0]+y_img[0,:,:,1]+y_img[0,:,:,2])/3
    ssim_loss_y = ssim_tf(fake_x_gray,y_img_gray)
#    ssim_loss_y =ssim_index_adjust( ssim_loss_y )
#    
#    ssim_loss = (ssim_loss_x+ssim_loss_y)/2
#    ssim_loss=2*(ssim_loss_1_x+ssim_loss_2_x+ssim_loss_3_x+ssim_loss_1_y+ssim_loss_2_y+ssim_loss_3_y)
    
#    store_x = tf.placeholder(tf.float32,shape=[10, 70, 70,3],name='store_image_x') #输入的x域图像
#    store_y = tf.placeholder(tf.float32,shape=[10, 70, 70,3],name='store_image_y') #输入的x域图像
    dy_fake = discriminator(image=fake_y, reuse=False, name='discriminator_y',buffer_use=False) #判别器返回的对生成的y域图像的判别结果
    dx_fake = discriminator(image=fake_x, reuse=False, name='discriminator_x',buffer_use=False) #判别器返回的对生成的x域图像的判别结果
    dy_real = discriminator(image=y_img, reuse=True, name='discriminator_y',buffer_use=False) #判别器返回的对真实的y域图像的判别结果
    dx_real = discriminator(image=x_img, reuse=True, name='discriminator_x',buffer_use=False) #判别器返回的对真实的x域图像的判别结果
 
#    gen_loss = gan_loss(dy_fake, tf.ones_like(dy_fake)) + gan_loss(dx_fake, tf.ones_like(dx_fake)) + args.lamda*l1_loss(x_img, fake_x_) + args.lamda*l1_loss(y_img, fake_y_) #计算生成器的loss
    
    L_cyc_loss = 1*l1_loss(x_img, fake_x_) + 1*l1_loss(y_img, fake_y_)
    
    G_loss = L_cyc_loss+gan_loss(dy_fake,tf.ones_like(dy_fake))+10*ssim_loss_x  ###train--G
    ##2 bu xing
    F_loss = L_cyc_loss+gan_loss(dx_fake,tf.ones_like(dx_fake))+10*ssim_loss_y ###train--F---->is right?????
    
    gen_loss_sum = G_loss+F_loss
    
    
    
    dy_loss_real = gan_loss(dy_real, tf.ones_like(dy_real)) #计算判别器判别的真实的y域图像的loss
    dy_loss_fake = gan_loss(dy_fake, tf.zeros_like(dy_fake)) #计算判别器判别的生成的y域图像的loss
    dy_loss =(dy_loss_real + dy_loss_fake) / 2 #计算判别器判别的y域图像的loss
 
    dx_loss_real = gan_loss(dx_real, tf.ones_like(dx_real)) #计算判别器判别的真实的x域图像的loss
    dx_loss_fake = gan_loss(dx_fake, tf.zeros_like(dx_fake)) #计算判别器判别的生成的x域图像的loss
    dx_loss =(dx_loss_real + dx_loss_fake) / 2 #计算判别器判别的x域图像的loss
    
    d_loss =dy_loss+dx_loss
#    lsof -i:6006
#    kill -9 19676
#    dis_loss = dy_loss + dx_loss #计算判别器的loss
    SSIM_loss_sum = tf.summary.scalar("final_objective_SSIM_x_to_y", ssim_loss_x)
#    color_loss_sum_G = tf.summary.scalar('color_loss_G',color_loss_G)
    G_loss_sum = tf.summary.scalar("final_objective_G", G_loss) #记录生成器loss的日志
    F_loss_sum = tf.summary.scalar("final_objective_F",  F_loss ) #记录生成器loss的日志
    Gen_loss_sum = tf.summary.scalar("final_objective_gen", gen_loss_sum)
#    
# 
    dx_loss_sum = tf.summary.scalar("dx_loss", dx_loss) #记录判别器判别的x域图像的loss的日志
    dy_loss_sum = tf.summary.scalar("dy_loss", dy_loss) #记录判别器判别的y域图像的loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", d_loss )#记录判别器的loss的日志
    discriminator_sum = tf.summary.merge([dx_loss_sum, dy_loss_sum, dis_loss_sum])
    
    

    
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph()) #日志记录器
 
#    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name] #所有生成器的可训练参数
#    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] #所有判别器的可训练参数
    G_vars = var('generator_x2y')
    F_vars = var('generator_y2x')
    dx_vars = var('discriminator_x')
    dy_vars = var('discriminator_y')
 
    lr = tf.placeholder(tf.float32, None, name='learning_rate') #训练中的学习率
    G_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1).minimize(G_loss,var_list=G_vars) #train--G
    F_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1).minimize(F_loss,var_list=F_vars) #train--F
    Dx_optim =  tf.train.AdamOptimizer(lr, beta1=args.beta1).minimize(dx_loss,var_list=dx_vars)##train--Dx
    Dy_optim =  tf.train.AdamOptimizer(lr, beta1=args.beta1).minimize(dy_loss,var_list=dy_vars)##train--Dy
#    s_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1).minimize(ssim_loss)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#    os.environ["CUDA_VISIBLE_ DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #新建会话层
    init = tf.global_variables_initializer() #参数初始化器
#   
    sess.run(init) #初始化所有可训练参数
#    check = os.listdir('/home/usrp1/djj_cycle_GAN/CycleGAN_color_correction/snapshots/')
#    saver = tf.train.import_meta_graph('/home/usrp1/djj_cycle_GAN/CycleGAN_color_correction/snapshots/cycle_model-159.meta')
    saver=tf.train.Saver()
#    saver.restore(sess, '/home/usrp1/djj_cycle_GAN/CycleGAN_color_correction/snapshots/cycle_model-159')
#    djj=0
    counter = 1 #counter记录训练步数
#    store_images_x = np.zeros((10,70,70,3),dtype=np.float32)
#    store_images_y = np.zeros((10,70,70,3),dtype=np.float32)
    for epoch in range(args.epoch): #训练epoch数
        shuffle(x_datalists) #每训练一个epoch，就打乱一下x域图像顺序
        shuffle(y_datalists) #每训练一个epoch，就打乱一下y域图像顺序
#        lrate = args.base_lr if epoch < args.epoch_step else args.base_lr*(args.epoch-epoch)/(args.epoch-args.epoch_step) #得到该训练epoch的学习率
        if counter<=50000:
            lrate = 0.0002
#        if (counter>10000 &counter<=40000):
#            lrate =0.00015
        if (counter>50000 &counter<=80000):
            lrate =0.0001
        if counter>=80000:
                lrate =0.00005
        for step in range(len(x_datalists)): #每个训练epoch中的训练step数
            counter += 1
            x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step, args.image_size) #读取x域图像和y域图像
            x_image_resize = white_img(x_image_resize)    
    #            batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0) #填充维度
#            batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0) #填充维度
            feed_dict = { lr : lrate, x_img : x_image_resize, y_img : y_image_resize} #得到feed_dict
#            fake_x_img,fake_x__img,fake_y_img,fake_y__img=sess.run([fake_x,fake_x_,fake_y,fake_y_], feed_dict=feed_dict)
#            feed_dict = { lr : lrate, input_fake_x:fake_x_img, input_fake_x_:fake_x__img, input_fake_y:fake_y_img, input_fake_y_:fake_y__img, input_x_img:x_image_resize, input_y_img:y_image_resize} #得到feed_dict
#            g_l,d_l = sess.run([gen_loss,dis_loss],feed_dict=feed_dict)
#            sess.run([d_optim, g_optim], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
            g_loss,f_loss,Dx_loss,Dy_loss,ssim_x,ssim_y,fake_x_value,fake_y_value=sess.run([G_loss, F_loss,dx_loss,dy_loss,ssim_loss_x,ssim_loss_y,fake_x,fake_y], feed_dict=feed_dict)
            
#            input_fake_y = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_y')
#    input_fake_y = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_y')
#    input_fake_x = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_x')
#    input_fake_y_ = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_y_')
#    input_fake_x_ = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_fake_x_')
#    input_x_img = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_x_img')
#    input_y_img = tf.tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='input_y_img')
                
            sess.run([G_optim, F_optim,Dx_optim,Dy_optim], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
            if counter % args.save_pred_every == 0: #每过save_pred_every次保存模型
#                save(saver, sess, args.snapshot_dir, counter)
                saver.save(sess,'./snapshots/cycle_model',global_step = counter)
            if counter % args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
#                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss, dis_loss], feed_dict=feed_dict)
                ssim_loss_value,Gen_loss_sum_value,dis_loss_sum_value=sess.run([SSIM_loss_sum,Gen_loss_sum,dis_loss_sum], feed_dict=feed_dict)
#                summary_writer.add_summary(color_loss_value_G, counter)
                summary_writer.add_summary(ssim_loss_value, counter)
                summary_writer.add_summary(Gen_loss_sum_value, counter)
                summary_writer.add_summary(dis_loss_sum_value, counter)
            if counter % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                fake_y_value, fake_x__value, fake_x_value, fake_y__value = sess.run([fake_y, fake_x_, fake_x, fake_y_], feed_dict=feed_dict) #run出网络输出
#                np.save('//home//usrp1//djj_cycle_GAN//CycleGAN_color_correction//train_out_num//'+'out'+str(counter)+'.npy',fake_y_value)
#                np.expand_dims(np.array(x_image_resize[0,:]).astype(np.float32), axis = 0)
                write_image = get_write_picture(x_image_resize, y_image_resize, fake_y_value, fake_x__value, fake_x_value, fake_y__value) #得到训练的可视化结果
                write_image_name = args.out_dir + "/out"+ str(counter) + ".png" #待保存的训练可视化结果路径与名称
                cv2.imwrite(write_image_name, write_image) #保存训练的可视化结果
#                djj = djj+1
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}, ssim_loss_x = {:.3f}'.format(epoch, step, (g_loss+f_loss), (Dx_loss+Dy_loss),ssim_x))
#            print('epoch {:d} step {:d} \t '.format(epoch, step))

if __name__ == '__main__':
    main()
