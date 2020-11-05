# -*- coding: utf-8 -*-
"""
Residual Attention Network for 3D Classification
"""

import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import UpSampling3D

from .basic_layers import ResidualBlock


class AttentionModule(object):
    """AttentionModuleClass"""
    def __init__(self, p=1, t=2, r=1):
        """
        :param p: the number of pre-processing Residual Units before splitting into trunk branch and mask branch
        :param t: the number of Residual Units in trunk branch
        :param r: the number of Residual Units between adjacent pooling layer in the mask branch
        """
        self.p = p
        self.t = t
        self.r = r

        self.residual_block = ResidualBlock()

    def f_prop(self, input, input_channels, scope="attention_module", is_training=True):
        """
        f_prop function of attention module
        :param input: A Tensor. input data [batch_size, depth,height, width, channel]
        :param input_channels: dimension of input channel.
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: A Tensor [batch_size, height, width, channel]
        """
        with tf.variable_scope(scope):

            # residual blocks
            with tf.variable_scope("first_residual_blocks"):
                for i in range(self.p):
                    input = self.residual_block.f_prop(input, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(self.t):
                    output_trunk = self.residual_block.f_prop(output_trunk, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("soft_mask_branch"):

                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = ( 3, 3, 3)
                    output_soft_mask = tf.layers.max_pooling3d(input,pool_size=filter_,strides=(2,2,2), padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.variable_scope("skip_connection"):
                    #
                    output_skip_connection = self.residual_block.f_prop(output_soft_mask, input_channels, is_training=is_training)


                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    filter_ = (3, 3, 3)
                    output_soft_mask =tf.layers.max_pooling3d(input,pool_size=filter_,strides=(2,2,2), padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)



                with tf.variable_scope("up_sampling_1"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling3D((1, 1,1))(output_soft_mask)

                # add skip connection   original skip output_skip_connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("up_sampling_2"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling3D((2,2,2))(output_soft_mask)



                with tf.variable_scope("output"):

                    #output_soft_mask=tf.layers.batch_normalization(output_soft_mask)
                    output_soft_mask = tf.layers.conv3d(output_soft_mask, filters=input_channels, kernel_size=(3,3,3),activation=tf.nn.relu ,padding="same")
                    #output_soft_mask = tf.layers.conv3d(output_soft_mask, filters=input_channels, kernel_size=(3,3,3),activation="relu",padding="same")


                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)
                    #output_soft_mask = tf.nn.softmax(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk
                #output = tf.add(output_trunk,output)

            with tf.variable_scope("last_residual_blocks"):
                for i in range(self.p):
                    output = self.residual_block.f_prop(output, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            return output