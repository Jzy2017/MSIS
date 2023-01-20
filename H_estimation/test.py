import os
import numpy as np
import cv2
import argparse
import tensorflow.compat.v1 as tf
from model import *
from data_utils_test import DataLoader

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_root', type=str, default='../example/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--device', type=str, default='0', help='device name if using cuda, else cpu')
args = parser.parse_args()

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

def makeroot(path):
	if not os.path.exists(path):
		os.makedirs(path)
def main():
	global_step = tf.train.get_or_create_global_step()

	genotypes_H=[('conv_5x5', 0), ('conv_5x5', 1), ('conv_1x1', 2),  
			  ('conv_3x3', 0), ('conv_5x5', 1), ('dil_conv_3x3', 2),
			  ('conv_5x5', 0), ('conv_5x5', 1), ('conv_3x3', 2), 
			  ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2)]
	
	test_data_loader = DataLoader(args.data_root)
	test_data_dataset = test_data_loader(batch_size=args.batch_size)
	test_data_iter = test_data_dataset.make_one_shot_iterator()
	(_, test_size_tensor) = test_data_iter.get_next()
	test_size_tensor.set_shape([args.batch_size, 2, 1])

	test_inputs_ir_1 = tf.placeholder(shape = [args.batch_size, None, None, 1], dtype = tf.float32)
	test_inputs_ir_2 = tf.placeholder(shape = [args.batch_size, None, None, 1], dtype = tf.float32)
	test_inputs_vis_1 = tf.placeholder(shape = [args.batch_size, None, None, 1], dtype = tf.float32)
	test_inputs_vis_2 = tf.placeholder(shape = [args.batch_size, None, None, 1], dtype = tf.float32)
	rgb1 = tf.placeholder(shape = [args.batch_size, None, None, 3], dtype = tf.float32)
	rgb2 = tf.placeholder(shape = [args.batch_size, None, None, 3], dtype = tf.float32)
	test_size_tensor = tf.placeholder(shape = [args.batch_size, 2, 1], dtype = tf.float32)

	logits,rgb = Model_rgb_test(test_inputs_ir_1, test_inputs_vis_1, test_inputs_ir_2, test_inputs_vis_2, rgb1, rgb2, genotypes_H, test_size_tensor, False)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	sess.run(tf.global_variables_initializer())
	restore_decoder_var = [v for v in tf.global_variables() if v.name.split('/')[0] == 'decoder' ]
	decoder_loader = tf.train.Saver(var_list = restore_decoder_var)
	decoder_loader.restore(sess, '../checkpoints/fusion_model/fusion_model')
	restore_var = [v for v in tf.global_variables() if v.name.split('/')[0] != 'decoder' ]
	loader = tf.train.Saver(var_list = restore_var)
	loader.restore(sess,'../checkpoints/H_model/H_model')

	for i in range(test_data_loader.get_length()):
		input_clip,input_rgb,filename =np.expand_dims( np.expand_dims(test_data_loader.get_rgb_data_clips(i, None, None)[0], axis=0), axis=3),\
								np.expand_dims(test_data_loader.get_rgb_data_clips(i, None, None)[1], axis=0),\
									test_data_loader.get_data_clips(i, None, None)[1]
		size_clip = np.expand_dims(test_data_loader.get_size_clips(i), axis=0)

		fuse, rgbs, _ = sess.run([ logits,rgb, global_step], feed_dict = {test_inputs_ir_1: input_clip[...,-4], test_inputs_vis_1: input_clip[...,-2],\
																	   test_inputs_ir_2: input_clip[...,-3], test_inputs_vis_2: input_clip[...,-1],\
																	   rgb1: input_rgb[...,:3], rgb2: input_rgb[...,3:6], test_size_tensor: size_clip})
		
		######  Color the fused image
		warp1 = (fuse[0,...,0])*255
		warp2 = (fuse[0,...,1])*255
		color1 = (rgbs[0,...,:3])*255
		color2 = (rgbs[0,...,3:6])*255
		vis_YCrCb_1 = cv2.cvtColor(color1, cv2.COLOR_BGR2YCrCb)
		vis_YCrCb_1[...,0] = warp1
		vis_warp1= cv2.cvtColor(vis_YCrCb_1, cv2.COLOR_YCrCb2BGR)
		rgb2_YCrCb = cv2.cvtColor(color2, cv2.COLOR_BGR2YCrCb)
		rgb2_YCrCb[...,0] = warp2
		vis_warp2 = cv2.cvtColor(rgb2_YCrCb, cv2.COLOR_YCrCb2BGR)

		makeroot(os.path.join('../results'))
		makeroot(os.path.join('../results', 'warp1'))
		makeroot(os.path.join('../results', 'warp2'))
		print(filename)
		cv2.imwrite(os.path.join('../results', 'warp1' , filename), vis_warp1)
		cv2.imwrite(os.path.join('../results', 'warp2' , filename), vis_warp2)

if __name__ == '__main__':
	main() 

