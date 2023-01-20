import os
import sys
import time 
import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tensorflow.compat.v1 as tf
import tflearn
import tf_slim as slim

OPS = {
  'conv_1x1' : lambda x, C, stride, batchnorm=True: Conv(x, C, [1,1], stride,batchnorm),
  'conv_3x3' : lambda x, C, stride, batchnorm=True: Conv(x, C, [3,3], stride,batchnorm),
  'conv_5x5' : lambda x, C, stride, batchnorm=True: Conv(x, C, [5,5], stride,batchnorm),
  'dil_conv_3x3' : lambda x, C, stride, batchnorm=True: DilConv(x, C, [3,3], stride,2,batchnorm),
  'dil_conv_5x5' : lambda x, C, stride, batchnorm=True: DilConv(x, C, [5,5], stride,2,batchnorm)
  }

def DilConv(x, C_out, kernel_size, stride, rate, batchnorm = 'True'):
	if batchnorm:
		x = tflearn.relu(x)
		C_in = x.shape[-1]
		x = slim.separable_convolution2d(x, C_out, kernel_size, depth_multiplier=1, stride = stride, rate = rate)
		x = slim.batch_norm(x)
	else:
		x = slim.conv2d(inputs = x, num_outputs = C_out, kernel_size = kernel_size, activation_fn = tf.nn.relu, rate = rate)
	return x
def Conv(x,C_out,kernel_size,stride,batchnorm = 'True'):
	if batchnorm:
		x = tflearn.relu(x)
		x = slim.separable_convolution2d(x, C_out, kernel_size, depth_multiplier = 1, stride = stride)
		x = slim.batch_norm(x)
	else:
		x = slim.conv2d(inputs = x, num_outputs = C_out ,kernel_size = kernel_size, activation_fn = tf.nn.relu)
	return x