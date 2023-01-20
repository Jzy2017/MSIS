import tensorflow.compat.v1 as tf
import tf_slim as slim
import tflearn
from operations import *

def Cell(s0,genotype, C_out, reduction,mode, change_channel=False):
	ops = OPS
	if mode == 'recon_encoder':
		op_names, indices = zip(*genotype.recon_en)
	elif mode == 'recon_encoder2':
		op_names, indices = zip(*genotype.recon_en2)
	elif mode == 'recon_decoder':
		op_names, indices = zip(*genotype.recon_de)
	elif mode == 'recon_decoder2':
		op_names, indices = zip(*genotype.recon_de2)
	cells_num = 4
	multiplier = 4
	s1 = ops[op_names[0]](s0, C_out, [1,1])
	state = [s0, s1]
	offset=0

	cells_num=4
	if reduction==True:
		cells_num+=1
	for i in range(cells_num-1):
		temp=[]
		for j in range(2):
			if reduction and indices[2*i+j+1]<2:
				stride = [2,2]# if reduction #and indices[2*i+j] < 2 else [1,1]
			else:
				stride = [1,1]
			h = state[indices[2*i+j+1]]
			temp.append(ops[op_names[2*i+j+1]](h, C_out, stride))   

		state.append(tf.add_n(temp))
	out = tf.concat(state[-multiplier:],axis=-1)
	return out


def Model_test(warp1, warp2, genotype):
	with tf.variable_scope('recon', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('recon_encoder', reuse = tf.AUTO_REUSE):
			en0 = slim.conv2d(tf.concat((warp1, warp2), -1), 64, [3,3], activation_fn = tflearn.relu)
			en1 = Cell(en0, genotype, 32, True, mode = 'recon_encoder')#64,64
			en2 = Cell(en1, genotype, 32, True, mode = 'recon_encoder2')#64,64
		with tf.variable_scope('recon_decoder', reuse = None): 
			de_up1 = slim.conv2d_transpose(inputs = en2, num_outputs = 256, kernel_size = 2, stride = 2) # 反卷积(长宽翻倍) 128->64
			de1 = Cell(tf.concat((de_up1, en1), -1), genotype, C_out = 32,change_channel = False, reduction = False, mode = 'recon_decoder')#,change_channel=True
			de_up2 = slim.conv2d_transpose(inputs = de1, num_outputs = 64, kernel_size = 2, stride = 2) # 反卷积(长宽翻倍) 128->64
			de2 = Cell(tf.concat((de_up2, en0),-1), genotype, C_out = 16,change_channel = False, reduction = False, mode = 'recon_decoder2')#,change_channel=True
			out = slim.conv2d(de2, 3, [3,3], activation_fn = None)
			out = tf.tanh(out)
	return out