import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
import genotypes 
from tensorDLT import solve_DLT
from tf_spatial_transform import transform
import output_tensorDLT
import output_tf_spatial_transform
import tflearn
from operations import *
from corr import vertical_attention, horizontal_attention

def MixedOp(x, op, C_out,stride,index,reduction,mode,change_channel=False):

	# time.sleep(3)
	out=OPS[op](x,C_out,stride)
	return out

def L1_norm(source_en_a, source_en_b):
	result = []
	resultb = []
	narry_a = source_en_a
	narry_b = source_en_b

	dimension = source_en_a.shape

	# caculate L1-norm
	temp_abs_a = tf.abs(narry_a)
	temp_abs_b = tf.abs(narry_b)
	_l1_a = tf.reduce_sum(temp_abs_a,3)
	_l1_b = tf.reduce_sum(temp_abs_b,3)

	_l1_a = tf.reduce_sum(_l1_a, 0)
	_l1_b = tf.reduce_sum(_l1_b, 0)

	l1_a = _l1_a#.eval()
	l1_b = _l1_b#.eval()

	# caculate the map for source images
	mask_value = l1_a + l1_b

	mask_sign_a = l1_a/mask_value
	mask_sign_b = l1_b/mask_value

	array_MASK_a = mask_sign_a
	array_MASK_b = mask_sign_b
	for b in range(dimension[0]):
		result=[]
		for i in range(dimension[3]):
			temp_matrix = array_MASK_a*narry_a[b,:,:,i] + array_MASK_b*narry_b[b,:,:,i]
			result.append(temp_matrix)
		results = tf.stack(result, axis=-1)
		resultb.append(tf.expand_dims(results,0))
	resultsb = tf.concat(resultb, axis=0)
	resule_tf =resultb
	return resule_tf
def Cell(s0,genotype, C_out, reduction,mode, change_channel=False):
	ops = OPS
	if mode == 'ir':
		op_names, indices = zip(*genotype.ir)
	elif mode == 'vis':
		op_names, indices = zip(*genotype.vis)
	elif mode == 'decoder':
		op_names, indices = zip(*genotype.decoder)
	cells_num = 4
	multiplier = 4
	s1 = ops[op_names[0]](s0, C_out, [1,1])
	state = [s0, s1]
	offset = 0
	cells_num = 4
	if reduction == True:
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

def vertical_cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [0, 0], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1
    cost_vol = []
    for y in range(0, max_offset):
        slice = tf.slice(padded_lvl, [0, y, 0, 0], [-1, h, -1, -1])
        cost = tf.reduce_mean(c1 * slice, axis = 3, keepdims = True)
        cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1)

    return cost_vol
def horizontal_cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [0, 0], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1
    cost_vol = []
    for x in range(0, max_offset):
      slice = tf.slice(padded_lvl, [0, 0, x, 0], [-1, -1, w, -1])
      cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
      cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1)
    return cost_vol
		


def Model_rgb_test(ir1, vis1, ir2, vis2,rgb1,rgb2, genotypes_H, size, is_training, cells_num = 4, multiplier = 4, stem_multiplier = 3, name = "model"):
	ir1_128 = tf.image.resize_images(ir1, [128,128],method=0)
	vis1_128 = tf.image.resize_images(vis1, [128,128],method=0)
	ir2_128 = tf.image.resize_images(ir2, [128,128],method=0)
	vis2_128 = tf.image.resize_images(vis2, [128,128],method=0)
	is_training=False
	batch_size = ir1.shape[0]
	with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
		# out_channel = 16
		genotype = eval("genotypes.%s" % 'DARTS_fusion')
		out_channel = 32
		with tf.variable_scope('ir_encoder', reuse=tf.AUTO_REUSE): 
			ir_en1 = slim.conv2d(ir1, out_channel,[3,3],activation_fn=tflearn.relu)
			ir_en1 = Cell(ir_en1, genotype,  out_channel, False, mode='ir')#64,64
		with tf.variable_scope('ir_encoder', reuse = tf.AUTO_REUSE): 
			ir_en2 = slim.conv2d(ir2, out_channel,[3,3],activation_fn=tflearn.relu)
			ir_en2 = Cell(ir_en2, genotype,  out_channel, False, mode='ir')#64,64

		with tf.variable_scope('ir_encoder', reuse=tf.AUTO_REUSE): 
			ir_en1_128 = slim.conv2d(ir1_128, out_channel,[3,3],activation_fn=tflearn.relu)
			ir_en1_128 = Cell(ir_en1_128, genotype,  out_channel, False, mode='ir')#64,64
		with tf.variable_scope('ir_encoder', reuse = tf.AUTO_REUSE): 
			ir_en2_128 = slim.conv2d(ir2_128, out_channel,[3,3],activation_fn=tflearn.relu)
			ir_en2_128 = Cell(ir_en2_128, genotype,  out_channel, False, mode='ir')#64,64

		with tf.variable_scope('vis_encoder', reuse=tf.AUTO_REUSE): 
			vis_en1 = slim.conv2d(vis1, out_channel,[3,3],activation_fn=tflearn.relu)
			vis_en1 = Cell(vis_en1, genotype, out_channel, False, mode='vis')
		with tf.variable_scope('vis_encoder', reuse = tf.AUTO_REUSE): 
			vis_en2 = slim.conv2d(vis2, out_channel,[3,3],activation_fn=tflearn.relu)
			vis_en2 = Cell(vis_en2, genotype, out_channel, False, mode='vis')

		with tf.variable_scope('vis_encoder', reuse=tf.AUTO_REUSE): 
			vis_en1_128 = slim.conv2d(vis1_128, out_channel,[3,3],activation_fn=tflearn.relu)
			vis_en1_128 = Cell(vis_en1_128, genotype, out_channel, False, mode='vis')
		with tf.variable_scope('vis_encoder', reuse = tf.AUTO_REUSE): 
			vis_en2_128 = slim.conv2d(vis2_128, out_channel,[3,3],activation_fn=tflearn.relu)
			vis_en2_128 = Cell(vis_en2_128, genotype, out_channel, False, mode='vis')
	with tf.variable_scope('H'): 
		with tf.variable_scope('fus_encoder', reuse=tf.AUTO_REUSE): 
			fus1_en = slim.max_pool2d(inputs=tf.concat((ir_en1_128, vis_en1_128),-1), kernel_size=2, padding='SAME') # 16,16,out
			feature1_1 = OPS[genotypes_H[0][0]](fus1_en,128,[1,1],False)
			feature1_1_5 = slim.max_pool2d(inputs = feature1_1, kernel_size = 2, padding = 'SAME') # 16,16,out
			feature1_2 = OPS[genotypes_H[1][0]](feature1_1_5, 128,[1,1],False)
			feature1_2_5 = slim.max_pool2d(inputs=feature1_2, kernel_size=2, padding='SAME') # 16,16,out
			feature1_3 = OPS[genotypes_H[2][0]](feature1_2_5, 128,[1,1],False)
		with tf.variable_scope('fus_encoder', reuse = True):
			fus2_en = slim.max_pool2d(inputs=tf.concat((ir_en2_128, vis_en2_128),-1), kernel_size=2, padding='SAME') # 16,16,out
			feature2_1 = OPS[genotypes_H[0][0]](fus2_en, 128,[1,1],False)
			feature2_1_5 = slim.max_pool2d(inputs=feature2_1, kernel_size=2, padding='SAME') # 16,16,out
			feature2_2 = OPS[genotypes_H[1][0]](feature2_1_5, 128,[1,1],False)
			feature2_2_5 = slim.max_pool2d(inputs=feature2_2, kernel_size=2, padding='SAME') # 16,16,out
			feature2_3 =  OPS[genotypes_H[2][0]](feature2_2_5, 128,[1,1],False)
		keep_prob = 0.5 if is_training==True else 1.0
		with tf.variable_scope('Reggression_Net1'): 
			search_range=16   
			vertical_attended_target = vertical_attention(tf.nn.l2_normalize(feature1_3,axis=3), tf.nn.l2_normalize(feature2_3,axis=3))     #垂直  attention
			horizontal_attended_target = horizontal_attention(tf.nn.l2_normalize(feature1_3,axis=3), tf.nn.l2_normalize(feature2_3,axis=3)) #水平  attention
			horizontal_correlation = horizontal_cost_volume(tf.nn.l2_normalize(feature1_3,axis=3),vertical_attended_target, search_range)               #水平 correlation
			vertical_correlation = vertical_cost_volume(tf.nn.l2_normalize(feature1_3,axis=3), horizontal_attended_target, search_range)               #垂直 correlation
			global_correlation = tf.concat([horizontal_correlation, vertical_correlation], axis=3)
			net1_flat = slim.flatten(global_correlation)
			# Two fully-connected layers
			with tf.variable_scope('net1_fc1'):
				net1_fc1 = slim.fully_connected(net1_flat, 1024, activation_fn=tf.nn.relu)
				net1_fc1 = slim.dropout(net1_fc1, keep_prob)
			with tf.variable_scope('net1_fc2'):
				net1_fc2 = slim.fully_connected(net1_fc1, 8, activation_fn=None) #BATCH_SIZE x 8
			net1_f = tf.expand_dims(net1_fc2, [2])
			patch_size = 32.
			H1 = solve_DLT(net1_f/4., patch_size)
			M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
						[0., patch_size / 2.0, patch_size / 2.0],
						[0., 0., 1.]]).astype(np.float32)
			M_tensor = tf.constant(M, tf.float32)
			M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
			M_inv = np.linalg.inv(M)
			M_tensor_inv = tf.constant(M_inv, tf.float32)
			M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
			H1 = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
			feature2_warp = transform(tf.nn.l2_normalize(feature2_2, axis=3), H1)
		with tf.variable_scope('Reggression_Net2'):    
			search_range=8
			vertical_attended_target = vertical_attention(tf.nn.l2_normalize(feature1_2,axis=3), tf.nn.l2_normalize(feature2_warp,axis=3))     #垂直  attention
			horizontal_attended_target = horizontal_attention(tf.nn.l2_normalize(feature1_2,axis=3), tf.nn.l2_normalize(feature2_warp,axis=3)) #水平  attention
			horizontal_correlation = horizontal_cost_volume(tf.nn.l2_normalize(feature1_2,axis=3),vertical_attended_target, search_range)               #水平 correlation
			vertical_correlation = vertical_cost_volume(tf.nn.l2_normalize(feature1_2,axis=3), horizontal_attended_target, search_range)               #垂直 correlation
			local_correlation_1 = tf.concat([horizontal_correlation, vertical_correlation], axis=3)						
			net2_flat = slim.flatten(local_correlation_1)
			# Two fully-connected layers
			with tf.variable_scope('net2_fc1'):
				net2_fc1 = slim.fully_connected(net2_flat, 512, activation_fn=tf.nn.relu)
				net2_fc1 = slim.dropout(net2_fc1, keep_prob)
			with tf.variable_scope('net2_fc2'):
				net2_fc2 = slim.fully_connected(net2_fc1, 8, activation_fn=None) #BATCH_SIZE x 8
			net2_f = tf.expand_dims(net2_fc2, [2])
			patch_size = 64.
			H2 = solve_DLT((net1_f+net2_f)/2., patch_size)
			M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
						[0., patch_size / 2.0, patch_size / 2.0],
						[0., 0., 1.]]).astype(np.float32)
			M_tensor = tf.constant(M, tf.float32)
			M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
			M_inv = np.linalg.inv(M)
			M_tensor_inv = tf.constant(M_inv, tf.float32)
			M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
			H2 = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
			feature3_warp = transform(tf.nn.l2_normalize(feature2_1,axis=3), H2)
		with tf.variable_scope('Reggression_Net3'):    
			search_range=8
			vertical_attended_target = vertical_attention(tf.nn.l2_normalize(feature1_1,axis=3), tf.nn.l2_normalize(feature3_warp,axis=3))     #垂直  attention
			horizontal_attended_target = horizontal_attention(tf.nn.l2_normalize(feature1_1,axis=3), tf.nn.l2_normalize(feature3_warp,axis=3)) #水平  attention
			horizontal_correlation = horizontal_cost_volume(tf.nn.l2_normalize(feature1_1,axis=3),vertical_attended_target, search_range)               #水平 correlation
			vertical_correlation = vertical_cost_volume(tf.nn.l2_normalize(feature1_1,axis=3), horizontal_attended_target, search_range)               #垂直 correlation
			local_correlation_2 = tf.concat([horizontal_correlation, vertical_correlation], axis=3)
			net3_flat = slim.flatten(local_correlation_2)
			# Two fully-connected layers
			with tf.variable_scope('net3_fc1'):
				net3_fc1 = slim.fully_connected(net3_flat, 256, activation_fn = tf.nn.relu)
				net3_fc1 = slim.dropout(net3_fc1, keep_prob)
			with tf.variable_scope('net3_fc2'):
				net3_fc2 = slim.fully_connected(net3_fc1, 8, activation_fn=None) #BATCH_SIZE x 8
			net3_f = tf.expand_dims(net3_fc2, [2])
		patch_size=128
		M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
					[0., patch_size / 2.0, patch_size / 2.0],
					[0., 0., 1.]]).astype(np.float32)
		M_tensor = tf.constant(M, tf.float32)
		M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
		# Inverse of M
		M_inv = np.linalg.inv(M)
		M_tensor_inv = tf.constant(M_inv, tf.float32)
		M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
		H3 = solve_DLT(net1_f+net2_f+net3_f, patch_size)
		H3_mat = tf.matmul(tf.matmul(M_tile_inv, H3), M_tile)

	with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE): 
		fus1 = slim.conv2d(tf.concat(L1_norm(ir_en1, vis_en1),0), 32,[3,3],activation_fn=tflearn.relu)
		fus1 = Cell(fus1, genotype,  C_out = 32,change_channel = False, reduction = False, mode = 'decoder')
		fus1 = slim.conv2d(fus1,1,[1,1])
		fus1 = tf.tanh(fus1)
	with tf.variable_scope('decoder', reuse = True): 
		fus2 = slim.conv2d(tf.concat(L1_norm(ir_en2, vis_en2),0), 32,[3,3],activation_fn=tflearn.relu)
		fus2 = Cell(fus2, genotype,  C_out = 32,change_channel = False, reduction = False, mode = 'decoder')
		fus2 = slim.conv2d(fus2,1,[1,1])
		fus2 = tf.tanh(fus2)

	shift = net1_f + net2_f + net3_f
	size_tmp = tf.concat([size,size,size,size],axis=1)/128.
	resized_shift = tf.multiply(shift, size_tmp)
	H = output_tensorDLT.solve_SizeDLT(resized_shift, size)  
	coarsealignment = output_tf_spatial_transform.Stitching_Domain_STN(tf.concat((fus1,fus2),-1), H, size, resized_shift)
	rgb = output_tf_spatial_transform.Stitching_Domain_STN(tf.concat((rgb1,rgb2),-1), H, size, resized_shift)

	return coarsealignment,rgb