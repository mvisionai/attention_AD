# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf
import numpy as np

from .basic_layers import ResidualBlock
from .attention_module import AttentionModule
from tensorflow.python.keras.layers import UpSampling3D


class ResidualAttentionNetwork(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        """
        :param input_shape: the list of input shape (ex: [None, 28, 28 ,3]
        :param output_dim:
        """

        self.output_dim = 2

        self.attention_module = AttentionModule()
        self.residual_block = ResidualBlock()

    def f_prop(self, x, is_training=True,keep_prop=1):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        # x = [None, row, line, channel]

        x = tf.layers.conv3d(x, filters=16, kernel_size=(6, 6, 6), strides=(3, 3, 3), padding='same', activation="relu",
                             use_bias=False)

        x = tf.layers.max_pooling3d(x, pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')


        x = self.attention_module.f_prop(x, input_channels=16, scope="attention_module_1", is_training=is_training)


        x = self.residual_block.f_prop(x, input_channels=16, output_channels=32, scope="residual_block_2",
                                       is_training=is_training)



        x = tf.layers.max_pooling3d(x, pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')

        x = self.attention_module.f_prop(x, input_channels=32, scope="attention_module_2", is_training=is_training)

        x = self.residual_block.f_prop(x, input_channels=32, output_channels=64, scope="residual_block_3",
                                       is_training=is_training)

        # x = self.attention_module.f_prop(x, input_channels=64, scope="attention_module_3", is_training=is_training)

        x = self.residual_block.f_prop(x, input_channels=64, output_channels=128, scope="residual_module_4",
                                       is_training=is_training)

        #x=tf.layers.conv3d(x, filters=170368, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                                 #name="cam_filter",use_bias=False)


        x = tf.layers.average_pooling3d(x, [3, 3, 3], (2, 2, 2), "same")


        print("shape average",x.get_shape())




        x = tf.keras.layers.Flatten()(x)

        print("shape ",x.get_shape().as_list()[1:])
        #x=tf.reshape(x, np.prod(x.get_shape().as_list()[1:]))


        total_collection = tf.add_to_collection("avg_collection", x)

        x = tf.layers.dense(x, 128, activation=tf.nn.relu)

        #x = tf.nn.dropout(x, keep_prop)

        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)





        return y




    def encoder(self,denoise_input,reuse=False):


        with tf.variable_scope(name_or_scope="Encoder",reuse=reuse):


            #print("Encoder ", denoise_input.get_shape())

            conv_1=tf.layers.conv3d(denoise_input, filters=16, kernel_size=(11,11,11), strides=(5,5,5),padding='same',activation="relu")

            bn_1=tf.layers.batch_normalization(conv_1)

            conv_2 = tf.layers.conv3d(bn_1, filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                      padding='same', activation="relu")

            max_pool_1 = tf.layers.max_pooling3d(conv_2, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

            #print("Encoder 2", max_pool_1.get_shape())

            bn_2 = tf.layers.batch_normalization(max_pool_1)

            conv_3 = tf.layers.conv3d(bn_2, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            bn_3 = tf.layers.batch_normalization(conv_3)
            conv_4 = tf.layers.conv3d(bn_3, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            max_pool_2 = tf.layers.max_pooling3d(conv_4, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')


            bn_4 = tf.layers.batch_normalization(max_pool_2)
            conv_5 = tf.layers.conv3d(bn_4, filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            bn_5 = tf.layers.batch_normalization(conv_5)
            conv_6 = tf.layers.conv3d(bn_5, filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu" )

            bn_6 = tf.layers.batch_normalization( conv_6 )
            conv_7 = tf.layers.conv3d( bn_6, filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            max_pool_3 = tf.layers.max_pooling3d(conv_7, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

            bn_7 = tf.layers.batch_normalization(max_pool_3)
            conv_8 = tf.layers.conv3d(bn_7, filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            bn_8 = tf.layers.batch_normalization( conv_8 )

            conv_9 = tf.layers.conv3d(bn_8, filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            bn_9 = tf.layers.batch_normalization(conv_9 )

            conv_10 = tf.layers.conv3d( bn_9 , filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")

            max_pool_4 = tf.layers.max_pooling3d(conv_10, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

            # conv_11 = tf.layers.conv3d(max_pool_4, filters=512, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',
            #                           activation="relu")
            # conv_12 = tf.layers.conv3d(conv_11, filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
            #                           activation="relu")
            # conv_13 = tf.layers.conv3d(conv_12, filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
            #                            activation="relu")
            #
            # max_pool_5 = tf.layers.max_pooling3d(conv_13, pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

            #print("Encoder " ,max_pool_4.get_shape())

        return max_pool_4


    def decoder(self,encoder_out):

        with tf.variable_scope(name_or_scope='Decoder') :


            #print("Encoder imm", encoder_out.get_shape())

            # dconv_1 =tf.layers.conv3d_transpose(inputs=encoder_out, filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            #                            activation="relu", padding="same")
            # dconv_2 =tf.layers.conv3d_transpose(dconv_1, filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
            #                           activation="relu", use_bias=False)
            #
            #
            #
            # dconv_3 = tf.layers.conv3d_transpose(dconv_2, filters=512, kernel_size=(3, 3, 3),strides=(2, 2, 2), padding='same',
            #                           activation="relu", use_bias=False)
            #
            # #dconv_3 = UpSampling3D(size=(2, 2, 2))(dconv_3)



            dconv_4 = tf.layers.conv3d_transpose(encoder_out, filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            bn_1 = tf.layers.batch_normalization( dconv_4)
            dconv_5 = tf.layers.conv3d_transpose(inputs=bn_1, filters=128, kernel_size=(3, 3, 3),
                                                 strides=(1, 1, 1),
                                                 activation="relu", padding="same")

            bn_2 = tf.layers.batch_normalization(dconv_5 )
            dconv_6 = tf.layers.conv3d_transpose(bn_2, filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            #dconv_6 = UpSampling3D(size=(2, 2, 2))(dconv_6)
            bn_3 = tf.layers.batch_normalization(dconv_6)

            dconv_7 = tf.layers.conv3d_transpose(bn_3 , filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            bn_4 = tf.layers.batch_normalization(dconv_7)
            dconv_8 = tf.layers.conv3d_transpose(bn_4, filters=64, kernel_size=(3, 3,3),strides=(1, 1, 1),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            bn_5 = tf.layers.batch_normalization(dconv_8)
            dconv_9 = tf.layers.conv3d_transpose(inputs= bn_5, filters=64, kernel_size=(3, 3, 3),
                                                 strides=(2, 2, 2),
                                                 activation="relu", padding="same")

            #dconv_9 = UpSampling3D(size=(2, 2, 2))(dconv_9)
            bn_6 = tf.layers.batch_normalization(dconv_9)
            #print("Decoder shape 1", dconv_9.get_shape())
            dconv_10 = tf.layers.conv3d_transpose(bn_6 , filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            bn_7 = tf.layers.batch_normalization(dconv_10)
            dconv_11 = tf.layers.conv3d_transpose(bn_7, filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            #dconv_11 = UpSampling3D(size=(3,3, 3))(dconv_11)

            bn_8 = tf.layers.batch_normalization(dconv_11)
            dconv_12 = tf.layers.conv3d_transpose(bn_8 , filters=16, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                                 padding='same',
                                                 activation="relu", use_bias=False)

            bn_9 = tf.layers.batch_normalization(dconv_12)
            dconv_13 = tf.layers.conv3d_transpose(inputs=bn_9, filters=1, kernel_size=(6, 6,6),
                                                  strides=(2, 2, 2),
                                                 activation="relu", padding="valid")

            # dconv_14 = tf.layers.conv3d_transpose(inputs=dconv_13, filters=1, kernel_size=(3, 3, 3),
            #                                       strides=(1, 1, 1),
            #                                       activation="relu", padding="same")

            #print("Decoder shape" ,dconv_13.get_shape())

        return   dconv_13


    def discriminator(self,z_proir,reuse=False):

        with tf.variable_scope(name_or_scope="Discriminator",reuse=reuse):

            conv_1 = tf.layers.conv3d(z_proir, filters=16, kernel_size=(11, 11, 11), strides=(3, 3, 3),
                                      padding='same', activation="relu")
            conv_2 = tf.layers.conv3d(conv_1, filters=16 ,kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                      padding='same', activation="relu")

            max_pool_1 = tf.layers.max_pooling3d(conv_2, pool_size=(3, 3, 3),  strides=(1, 1, 1), padding='same')

            conv_3 = tf.layers.conv3d(max_pool_1, filters=32, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                      activation="relu")
            conv_4 = tf.layers.conv3d(conv_3, filters=32, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                      activation="relu")

            max_pool_2 = tf.layers.max_pooling3d(conv_4, pool_size=(3, 3, 3),  strides=(1, 1, 1), padding='same')

            conv_5 = tf.layers.conv3d(max_pool_2, filters=64, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                      activation="relu")
            conv_6 = tf.layers.conv3d(conv_5, filters=64, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                      activation="relu")
            conv_7 = tf.layers.conv3d(conv_6, filters=64, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                      activation="relu")

            max_pool_3 = tf.layers.max_pooling3d(conv_7, pool_size=(3, 3, 3),  strides=(1, 1, 1), padding='same')

            conv_8 = tf.layers.conv3d(max_pool_3, filters=128, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                      activation="relu")
            conv_9 = tf.layers.conv3d(conv_8, filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                      activation="relu")
            conv_10 = tf.layers.conv3d(conv_9, filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                       activation="relu")

            max_pool_4 = tf.layers.max_pooling3d(conv_10, pool_size=(3, 3, 3),  strides=(1, 1, 1), padding='same')

            conv_11 = tf.layers.conv3d(max_pool_4, filters=256, kernel_size=(3, 3, 3),  strides=(1, 1, 1),
                                       padding='same',
                                       activation="relu")
            conv_12 = tf.layers.conv3d(conv_11, filters=256, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                       activation="relu")
            conv_13 = tf.layers.conv3d(conv_12, filters=256, kernel_size=(3, 3, 3),  strides=(1, 1, 1), padding='same',
                                       activation="relu")

            max_pool_5 = tf.layers.max_pooling3d(conv_13, pool_size=(3, 3, 3),  strides=(1, 1, 1), padding='same')

            dense_layer = tf.layers.dense(max_pool_5,1)

            discriminator_output = tf.nn.sigmoid(dense_layer)



        return discriminator_output