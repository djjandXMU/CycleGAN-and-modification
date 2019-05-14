# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:24:14 2018

@author: djj
"""

import numpy as np
import tensorflow as tf
import math
import random
#构造可训练参数
def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)
 
#定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = True):
    input_dim = input_.get_shape()[-1]
#    with tf.device('/gpu:1'):
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 
#定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 
#定义反卷积层
def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output
 
#定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output
 
#定义最大池化层
def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
 
#定义平均池化层
def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
 
#定义lrelu激活层
def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)
 
#定义relu激活层
def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)
 
#定义残差块
def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    else:
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    add_raw = input_ + conv2dc1_norm
    output = relu(input_ = add_raw)
    return output
 
#定义生成器
def generator(image, gf_dim=64, reuse=False, name="generator"): 
    #生成器输入尺度: 1*256*256*3  
    input_dim = image.get_shape()[-1]
#    with tf.device('/gpu:1'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        #第1个卷积模块，输出尺度: 1*256*256*64  
        c0 = relu(batch_norm(conv2d(input_ = image, output_dim = gf_dim, kernel_size = 7, stride = 1, name = 'g_e0_c'), name = 'g_e0_bn'))
        #第2个卷积模块，输出尺度: 1*128*128*128
        c1 = relu(batch_norm(conv2d(input_ = c0, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_e1_c'), name = 'g_e1_bn'))
        #第3个卷积模块，输出尺度: 1*64*64*256
        c2 = relu(batch_norm(conv2d(input_ = c1, output_dim = gf_dim * 4, kernel_size = 3, stride = 2, name = 'g_e2_c'), name = 'g_e2_bn'))
        
        #9个残差块:
        r1 = residule_block_33(input_ = c2, output_dim = gf_dim*4, atrous = False, name='g_r1')
        r2 = residule_block_33(input_ = r1, output_dim = gf_dim*4, atrous = False, name='g_r2')
        r3 = residule_block_33(input_ = r2, output_dim = gf_dim*4, atrous = False, name='g_r3')
        r4 = residule_block_33(input_ = r3, output_dim = gf_dim*4, atrous = False, name='g_r4')
        r5 = residule_block_33(input_ = r4, output_dim = gf_dim*4, atrous = False, name='g_r5')
        r6 = residule_block_33(input_ = r5, output_dim = gf_dim*4, atrous = False, name='g_r6')
        r7 = residule_block_33(input_ = r6, output_dim = gf_dim*4, atrous = False, name='g_r7')
        r8 = residule_block_33(input_ = r7, output_dim = gf_dim*4, atrous = False, name='g_r8')
        r9 = residule_block_33(input_ = r8, output_dim = gf_dim*4, atrous = False, name='g_r9')
        #第9个残差块的输出尺度: 5*64*64*256
 
		#第1个反卷积模块，输出尺度: 1*128*128*128
        d1 = relu(batch_norm(deconv2d(input_ = r9, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_d1_dc'),name = 'g_d1_bn'))
		#第2个反卷积模块，输出尺度: 1*256*256*64
        d2 = relu(batch_norm(deconv2d(input_ = d1, output_dim = gf_dim, kernel_size = 3, stride = 2, name = 'g_d2_dc'),name = 'g_d2_bn'))
#        d3 = relu(batch_norm(deconv2d(input_ = d2, output_dim = gf_dim, kernel_size = 3, stride = 2, name = 'g_d3_dc'),name = 'g_d3_bn'))
#        d4 = relu(batch_norm(deconv2d(input_ = d3, output_dim = gf_dim, kernel_size = 3, stride = 2, name = 'g_d4_dc'),name = 'g_d4_bn'))
        #最后一个卷积模块，输出尺度: 1*256*256*3
        d3 = conv2d(input_=d2, output_dim  = input_dim, kernel_size = 7, stride = 1, name = 'g_d3_c')
        d4 = conv2d(input_=d3, output_dim  = input_dim, kernel_size = 7, stride = 1, name = 'g_d4_c')
        d5 = conv2d(input_=d4, output_dim  = input_dim, kernel_size = 7, stride = 1, name = 'g_d5_c')
#        d3 = conv2d(input_=d2, output_dim  = input_dim, kernel_size = 7, stride = 1, name = 'g_d4_c')
#        d3 = conv2d(input_=d2, output_dim  = input_dim, kernel_size = 7, stride = 1, name = 'g_d5_c')
		#经过tanh函数激活得到生成的输出
        output = tf.nn.tanh(d5)
        return output
 
#定义判别器,buffer is used to store previous images which were generated in the certain domain.
def discriminator(image, df_dim=64, reuse=False, name="discriminator",buffer_use=False,buffer = None):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        if buffer_use==False:
#            with tf.device('/gpu:0'):
            image_crop_1 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_1')
            image_crop_2 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_2')
            image_crop_3 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_3')
            image_crop_4 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_4')
            image_crop_5 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_5')
            image_crop_6 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_6')
            image_crop_7 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_7')
            image_crop_8 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_8')
            image_crop_9 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_9')
            image_crop_10 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_10')
            image_crop_11 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_11')
            image_crop_12 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_12')
            image_crop_13 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_13')
            image_crop_14 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_14')
            image_crop_15 = tf.random_crop(image, [1, 30, 30, 3],name='image_crop_15')
            image_ensemble = tf.concat([image_crop_1,image_crop_2,image_crop_3,image_crop_4,image_crop_5,image_crop_6,image_crop_7,image_crop_8,image_crop_9,image_crop_10,image_crop_11,image_crop_12,image_crop_13,image_crop_14,image_crop_15],axis=0)
            		#第1个卷积模块，输出尺度: 1*128*128*64
            h0 = lrelu(conv2d(input_ = image_ensemble, output_dim = df_dim, kernel_size = 4, stride = 2, name='d_h0_conv'))
            		#第2个卷积模块，输出尺度: 1*64*64*128
            h1 = lrelu(batch_norm(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 4, stride = 2, name='d_h1_conv'), 'd_bn1'))
            		#第3个卷积模块，输出尺度: 1*32*32*256
            h2 = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 4, stride = 2, name='d_h2_conv'), 'd_bn2'))
            		#第4个卷积模块，输出尺度: 1*32*32*512
            h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 2, name='d_h3_conv'), 'd_bn3'))
            		#最后一个卷积模块，输出尺度: 1*32*32*1
            output = conv2d(input_ = h3, output_dim = 1, kernel_size = 4, stride = 1, name='d_h4_conv')
        else:
            rand_seed = random.sample(range(15),13)
            for i in range(13):
                tf.assign(buffer[rand_seed[i],:,:,:],tf.random_crop(image, [1, 70, 70, 3]))
            
#            image_crop_1 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_1')
#            image_crop_2 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_2')
#            image_crop_3 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_3')
#            image_crop_4 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_4')
#            image_crop_5 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_5')
#            image_crop_6 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_6')
#            image_crop_7 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_7')
#            image_crop_8 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_8')
#            image_crop_9 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_9')
#            image_crop_10 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_10')
#            image_crop_11 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_11')
#            image_crop_12 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_12')
#            image_crop_13 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_13')
#            image_crop_14 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_14')
#            image_crop_15 = tf.random_crop(image, [1, 70, 70, 3],name='image_crop_15')
#            image_ensemble = tf.concat([image_crop_1,image_crop_2,image_crop_3,image_crop_4,image_crop_5,image_crop_6,image_crop_7,image_crop_8,image_crop_9,image_crop_10,image_crop_11,image_crop_12,image_crop_13,image_crop_14,image_crop_15],axis=0)
##            random_seed =random.sample(range(0,20),20)
#            for i in range(1):
#                tf.assign(buffer[random_seed[i],:,:,:],tf.random_crop(image, [1, 70, 70, 3],seed=None))
#            		第1个卷积模块，输出尺度: 1*128*128*64
            h0 = lrelu(conv2d(input_ = buffer, output_dim = df_dim, kernel_size = 4, stride = 2, name='d_h0_conv'))
            		#第2个卷积模块，输出尺度: 1*64*64*128
            h1 = lrelu(batch_norm(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 4, stride = 2, name='d_h1_conv'), 'd_bn1'))
            		#第3个卷积模块，输出尺度: 1*32*32*256
            h2 = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 4, stride = 2, name='d_h2_conv'), 'd_bn2'))
            		#第4个卷积模块，输出尺度: 1*32*32*512
            h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 2, name='d_h3_conv'), 'd_bn3'))
            		#最后一个卷积模块，输出尺度: 1*32*32*1
            output = conv2d(input_ = h3, output_dim = 1, kernel_size = 4, stride = 1, name='d_h4_conv')
        return output
