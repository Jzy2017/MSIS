import tensorflow.compat.v1 as tf
from tf_slim import conv2d
import tf_slim as slim

class PositionEmbeddingLearned(object):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.initializer = tf.random_uniform_initializer(0, 1)
        self.row_embed = tf.keras.layers.Embedding(128, num_pos_feats, embeddings_initializer=tf.random_uniform_initializer(0, 1))
        self.col_embed = tf.keras.layers.Embedding(128, num_pos_feats, embeddings_initializer=tf.random_uniform_initializer(0, 1))

    def __call__(self, tensor_list):
        # [B, H, W, C]
        x = tensor_list
        # h, w = x.shape[-2:]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        i = tf.cast(tf.range(w), tf.int64)
        j = tf.cast(tf.range(h), tf.int64)
        print('input: ', i)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = tf.concat([
            tf.tile(tf.expand_dims(x_emb, [0]), tf.stack([h, 1, 1])),
            tf.tile(tf.expand_dims(y_emb, [1]), tf.stack([1, w, 1])),
        ], axis=-1)
        pos = tf.expand_dims(pos, [0])
        pos = tf.tile(pos, tf.stack([tf.shape(x)[0], 1, 1, 1]))
        # [B, H, W, C]
        return pos



def vertical_attention(source, target):
    # [B, H, W, C]
    # _, height, width, channel = tf.shape(source)
    height = tf.shape(source)[1]
    width = tf.shape(source)[2]
    channel = source.get_shape().as_list()[3]
    print('channel: ', channel)
    pos_encoding = PositionEmbeddingLearned(channel//2)
    source_pos = pos_encoding(source) + source
    print('positioned: ', source_pos.shape)
    # compute horizontal self-attention
    with tf.variable_scope('vertical_conv_1'):
        source_pos_1 =  slim.conv2d(inputs=source_pos, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
        source_pos_2 =  slim.conv2d(inputs=source_pos, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
    source_pos_2 = tf.transpose(source_pos_2, [0, 1, 3, 2]) # [B, H, C, W]
    attention_weights = tf.nn.softmax(tf.matmul(source_pos_1, source_pos_2), axis=3) # [B, H, W, W]
    source_attended = tf.matmul(attention_weights, source) # [B, H, W, C]

    # compute vertical cross attention
    with tf.variable_scope('vertical_conv_2'):
        source_pos_v =  slim.conv2d(inputs=pos_encoding(source_attended) + source_attended, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
        target_pos_v =  slim.conv2d(inputs=pos_encoding(target) + target, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
    source_pos_v = tf.transpose(source_pos_v, [0, 2, 1, 3]) # [B, W, H, C]
    target_pos_v = tf.transpose(target_pos_v, [0, 2, 3, 1]) # [B, W, C, H]
    attention_weights_v = tf.nn.softmax(tf.matmul(source_pos_v, target_pos_v), axis=3) # [B, W, H, H]
    attention_res = tf.matmul(attention_weights_v, tf.transpose(target, [0, 2, 1, 3])) # [B, W, H, C]

    # attention_res, source

    return tf.transpose(attention_res, [0, 2, 1, 3])

def horizontal_attention(source, target):
    # [B, H, W, C]
    height = tf.shape(source)[1]
    width = tf.shape(source)[2]
    channel = source.get_shape().as_list()[3]
    pos_encoding = PositionEmbeddingLearned(channel//2)
    source_pos = pos_encoding(source) + source

    # compute vertical self-attention
    with tf.variable_scope('horizontal_conv_1'):
        source_pos_1 =  slim.conv2d(inputs=source_pos, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
        source_pos_2 =  slim.conv2d(inputs=source_pos, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
    source_pos_1 = tf.transpose(source_pos_1, [0, 2, 1, 3]) # [B, W, H, C]
    source_pos_2 = tf.transpose(source_pos_2, [0, 2, 3, 1])  # [B, W, C, H]
    attention_weights = tf.nn.softmax(tf.matmul(source_pos_1, source_pos_2), axis=3) # [B, W, H, H]
    source_attented = tf.matmul(attention_weights, tf.transpose(source, [0, 2, 1, 3])) # [B, W, H, C]
    source_attented = tf.transpose(source_attented, [0, 2, 1, 3]) # [B, H, W, C]

    # compute horizontal cross attention
    with tf.variable_scope('horizontal_conv_2'):
        source_pos_h =  slim.conv2d(inputs=pos_encoding(source_attented) + source_attented, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
        target_pos = slim.conv2d(inputs=pos_encoding(target) + target, num_outputs=channel, kernel_size=1, padding='valid', activation_fn=tf.nn.relu)
    target_pos = tf.transpose(target_pos, [0, 1, 3, 2]) # [B, H, C, W]
    attention_weights_v = tf.nn.softmax(tf.matmul(source_pos_h, target_pos), axis=3)  # [B, H, W, W]
    attention_res = tf.matmul(attention_weights_v, target) # [B, H, W, C]

    return  attention_res
