import tensorflow.compat.v1 as tf
import tflearn
import tf_slim as slim


OPS = {
  'conv_1x1' : lambda x, C, stride: Conv(x, C, [1,1], stride),
  'conv_3x3' : lambda x, C, stride: Conv(x, C, [3,3], stride),
  'conv_5x5' : lambda x, C, stride: Conv(x, C, [5,5], stride),
  'dil_conv_3x3' : lambda x, C, stride: DilConv(x, C, [3,3], stride, 2),
  'dil_conv_5x5' : lambda x, C, stride: DilConv(x, C, [5,5], stride, 2)
  }

def Zero(x,stride):
	return tf.zeros_like(x)[:,::stride[0],::stride[1],:]

def DilConv(x,C_out,kernel_size,stride,rate):
	
  x=slim.separable_convolution2d(x,C_out,kernel_size,depth_multiplier=1,stride=stride,rate=rate)
  x=tflearn.relu(x)
  # x=slim.batch_norm(x)
  return x

def Conv(x,C_out,kernel_size,stride):
  # x=tflearn.relu(x)
  x=slim.separable_convolution2d(x,C_out,kernel_size,depth_multiplier=1,stride=stride)
  x=tflearn.relu(x)
  # x=slim.batch_norm(x)
  return x

