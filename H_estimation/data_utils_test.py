import tensorflow.compat.v1 as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import time
from tf_slim import conv2d
import tf_slim as slim
rng = np.random.RandomState(2022)

# test: get input gray images
def np_load_frame(filename, resize_height, resize_width, interpolation = cv2.INTER_LINEAR):
    image_decoded = cv2.imread(filename)
 
    if resize_height != None and resize_width != None:
        image_resized = cv2.resize(image_decoded, (resize_width, resize_height), interpolation=interpolation)
    else:
        image_resized = image_decoded
    image_resized = np.expand_dims(cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY), 2)
    image_resized = image_resized.astype(dtype = np.float32)
    image_resized = image_resized/255.
    return image_resized

# test: get input rgb images
def np_load_frame_rgb(filename, resize_height, resize_width, interpolation = cv2.INTER_LINEAR):
    image_decoded = cv2.imread(filename)
    if resize_height != None and resize_width != None:
        image_resized = cv2.resize(image_decoded, (resize_width, resize_height), interpolation=interpolation)
    else:
        image_resized = image_decoded
    image_resized = image_resized.astype(dtype = np.float32)
    image_resized = image_resized/255.
    return image_resized

# test: get size of input images
def np_load_size(filename):
    image_decoded = cv2.imread(filename)
    height = image_decoded.shape[0]
    width = image_decoded.shape[1]
    size = np.array([width, height], dtype=np.float32)
    return np.expand_dims(size, 1)

class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.setup(data_folder)

    def __call__(self, batch_size):
        data_info_list = list(self.datas.values())
        length = data_info_list[0]['length']

        def data_clip_generator():
            while True:
                data_clip = []
                size_clip = []
                shift_clip = []
                frame_id = rng.randint(0, length-1)
                data_clip.append(np_load_frame(data_info_list[0]['frame'][frame_id], 128, 128))
                data_clip.append(np_load_frame(data_info_list[1]['frame'][frame_id], 128, 128))
                data_clip.append(np_load_frame(data_info_list[2]['frame'][frame_id], 128, 128))
                data_clip.append(np_load_frame(data_info_list[3]['frame'][frame_id], 128, 128))
                data_clip = np.concatenate(data_clip, axis=2)
                size_clip.append(np_load_size(data_info_list[0]['frame'][frame_id]))
                size_clip = np.concatenate(size_clip, axis=0)
                yield (data_clip, size_clip)

        dataset = tf.data.Dataset.from_generator(generator=data_clip_generator, output_types=(tf.float32, tf.float32),
                                                  output_shapes=([128, 128, 4], [2,1]))
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=60)
        dataset = dataset.shuffle(buffer_size=60).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, data_name):
        assert data_name in self.datas.keys(), 'data = {} is not in {}!'.format(data_name, self.datas.keys())
        return self.datas[data_name]

    def setup(self,data_folder):
        datas = glob.glob(os.path.join(self.dir, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'ir_input1' or data_name == 'ir_input2' or data_name == 'vis_input1' or data_name == 'vis_input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['frame'] = glob.glob(os.path.join(data, '*.png'))
                self.datas[data_name]['frame'].sort()
                self.datas[data_name]['length'] = len(self.datas[data_name]['frame'])
        print('data keys: ', self.datas.keys())

    def get_data_clips(self, index, resize_height, resize_width):
        batch = []
        data_info_list = list(self.datas.values())
        filename = data_info_list[0]['frame'][index].split('/')[-1]
        for i in [0,1,2,3]:
            image = np_load_frame(data_info_list[i]['frame'][index], None, None)
            batch.append(image)
        return np.concatenate(batch, axis=2),filename
    def get_rgb_data_clips(self, index, resize_height, resize_width):
        batch = []
        batch_rgb = []
        data_info_list = list(self.datas.values())
        filename = data_info_list[0]['frame'][index].split('/')[-1]
        for i in [0,1,2,3]:
            image = np_load_frame(data_info_list[i]['frame'][index], None, None)
            batch.append(image)
        for i in [2,3]:
            image = np_load_frame_rgb(data_info_list[i]['frame'][index], None, None)
            batch_rgb.append(image)
        return np.concatenate(batch, axis=2),np.concatenate(batch_rgb,2),filename

    # test: get size
    def get_size_clips(self, index):
        batch = []
        data_info_list = list(self.datas.values())
        size = np_load_size(data_info_list[0]['frame'][index])
        return size
        
    def get_length(self):
        return len(list(self.datas.values())[0]['frame'])
