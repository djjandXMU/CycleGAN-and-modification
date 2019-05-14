# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:22:55 2018

@author: djj
"""
import cv2
import tensorflow as tf
import numpy as np

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
#        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                    (sigma1_sq + sigma2_sq + C2))
        value=tf.div(tf.multiply(tf.add(tf.multiply(2,mu1_mu2),C1),tf.add(tf.multiply(2,sigma12),C2)),tf.multiply(tf.add(tf.add(mu1_sq,mu2_sq),C1),tf.add(tf.add(sigma1_sq,sigma2_sq),C2)))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
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

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

x_img = tf.placeholder(tf.float32,shape=[1, 256, 256,1],name='x_img') #输入的x域图像,n*col*row*channel
y_img = tf.placeholder(tf.float32,shape=[1, 256, 256,1],name='y_img') #输入的y域图像,n*col*row*channel

tf_ssim_loss_ms = tf_ms_ssim(x_img, y_img, mean_metric=True, level=5)
tf_ssim_loss = tf_ssim(x_img, y_img, cs_map=False, mean_metric=True, size=11, sigma=1.5)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #设定显存不超量使用
sess = tf.Session(config=config) #新建会话层
init = tf.global_variables_initializer() #参数初始化器
sess.run(init) #初始化所有可训练参数




img1_gray = np.zeros((1,256,256,1))
img2_gray = np.zeros((1,256,256,1))

img1 = np.float32(cv2.imread('//home//usrp1//djj_cycle_GAN//img//tongue_type_1//type_1_standard_msr//image_1.jpg'))
img1 =cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1 = (cv2.resize(img1, (256, 256),interpolation=cv2.INTER_AREA)) #改变读取的x域图片的大小
img1_gray[0,:,:,0] = img1
#img1 =cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#img1 = (cv2.resize(img1, (256, 256),interpolation=cv2.INTER_AREA))/255 #改变读取的x域图片的大小
#img1_gray[0,:,:,0] = img1
#img1 = np.expand_dims(np.array(img1).astype(np.float32), axis = 0) #填充维度

img2 = np.float32(cv2.imread('//home//usrp1//djj_cycle_GAN//img//tongue_type_1//type_1_standard_msr//image_1.jpg'))
img2 =cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img2 = (cv2.resize(img2, (256, 256),interpolation=cv2.INTER_AREA)) #改变读取的x域图片的大小
img2_gray[0,:,:,0] = img2

#img2 = np.expand_dims(np.array(img2).astype(np.float32), axis = 0) #填充维度
feed_dict={x_img:img1_gray,y_img:img2_gray}
##
ssim_value_ms,ssim_loss = sess.run([tf_ssim_loss_ms,tf_ssim_loss],feed_dict = feed_dict)
