import os
import numpy as np
import cv2
import argparse
import tensorflow.compat.v1 as tf
from model import *
from data import * 
import genotypes 

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_root', type = str, default = '../results', help='location of the data corpus')
parser.add_argument('--batch_size', type = int, default = 1, help='batch size')
parser.add_argument('--device', type = str, default = '2', help='device name if using cuda, else ''cpu'' ')
args = parser.parse_args()
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

def makeroot(path):
	if not os.path.exists(path):
		os.makedirs(path)
def main():
	global_step = tf.train.get_or_create_global_step()
	genotype = eval("genotypes.%s" % 'DARTS_stitch')
	test_data_loader = DataLoader(args.data_root)
	train_inputs_warp1 = tf.placeholder(shape=[args.batch_size, None, None,3], dtype=tf.float32)
	train_inputs_warp2 = tf.placeholder(shape=[args.batch_size, None, None,3], dtype=tf.float32)
	logits = Model_test(train_inputs_warp1, train_inputs_warp2, genotype)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	saver = tf.train.Saver(max_to_keep=1)
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, '../checkpoints/reconstruction_model/reconstruction_model')


	_infer_loss=0
	makeroot(os.path.join('../results', 'recon'))
	for i in range(test_data_loader.get_length()):
		input_data, name = test_data_loader.get_data_clips(i, None, None)
		print(name)
		input_clip = np.expand_dims(input_data, axis=0)
		fuse, _ = sess.run([ logits, global_step],feed_dict = {train_inputs_warp1: input_clip[...,:3], train_inputs_warp2: input_clip[...,3:]})
		cv2.imwrite(os.path.join('../results/recon/',name), fuse[0]*255)

if __name__ == '__main__':
	main() 

