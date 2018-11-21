#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:22:13 2017

@author: no1
"""

import tensorflow as tf

def stem(name, inputs):
  with tf.variable_scope(name) :
    #inputs (batch,299,299,3)
    #-------------------------------------------------------------------
    layers1 = tf.layers.conv2d(inputs, filters = 32, kernel_size = 3, strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers1') #(batch,149,149,32)
    #-------------------------------------------------------------------
    layers2 = tf.layers.conv2d(layers1, filters = 32, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers2') #(batch,147,147,32)

    #-------------------------------------------------------------------
    layers3 = tf.layers.conv2d(layers2, filters = 64, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3')#(batch,147,147,32)
    #---------------------------------------------------------------------
    #right
    layers4 = tf.layers.conv2d(layers3, filters = 96, kernel_size = 3, strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers3_1')#(30, 73, 73, 96)

    #left
    pool1 = tf.nn.max_pool(layers3, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID', name = name +'_pool1')#(30, 73, 73, 64)

    #concat
    layers5 = tf.concat([pool1, layers4], axis = -1)#(30, 73, 73, 160)

    #-----------------------------------------------------------
    #right line
    layers6_1 = tf.layers.conv2d(layers5, filters = 64, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers5_1')#(30, 73, 73, 64)

    layers6_2 = tf.layers.conv2d(layers6_1, filters = 64, kernel_size =[7, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers5_2')#(30, 73, 73, 64)

    layers6_3 = tf.layers.conv2d(layers6_2, filters = 64, kernel_size =[1, 7], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers5_3')#(30, 73, 73, 64)

    layers6_4 = tf.layers.conv2d(layers6_3, filters = 96, kernel_size =[3, 3], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers5_4')#(30, 71, 71, 96)

    #left line
    layers6_1_1 = tf.layers.conv2d(layers5, filters = 64, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers6_1_1')#(30, 73, 73, 64)

    layers6_2_1 = tf.layers.conv2d(layers6_1_1, filters = 96, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers6_2_1') #(30, 73, 73, 96)

    #concat

    layers7 = tf.concat([layers6_2_1, layers6_4], axis = -1, name = name + '_layers7')   #(batch,71,71,192)


    #--------------------------------------------------------------
    #right line
    pool2 = tf.nn.max_pool(layers7, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID', name = name +'_pool2')#(30, 35, 35, 192)
    #left line
    layers8 = tf.layers.conv2d(layers7, filters = 192, kernel_size = 3, strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers8')
    #concat
    layers9 = tf.concat([layers8, pool2], axis = -1)#(30, 35, 35, 384)

    return layers9


def inception_a(name, inputs):
  with tf.variable_scope(name):
    #---------------------------------------------------------------------------------
    #line1
    pool1 = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME', name = name +'_pool1')#(30, 35, 35, 384)

    layers1 = tf.layers.conv2d(pool1, filters = 96, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers1')#(30, 35, 35, 384)

    #----------------------------------------------------------------------------------
    #line2
    layers2 = tf.layers.conv2d(inputs, filters = 96, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2')#(30, 35, 35, 96)

    #----------------------------------------------------------------------------------
    #line3
    layers3_1 = tf.layers.conv2d(inputs, filters = 64, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_1')
    #line3
    layers3_2 = tf.layers.conv2d(layers3_1, filters = 96, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_2')#(30, 35, 35, 96)

    #----------------------------------------------------------------------------------
    #line4
    layers4_1 = tf.layers.conv2d(inputs, filters = 64, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_1')

    layers4_2 = tf.layers.conv2d(layers4_1, filters = 96, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_2')

    layers4_3 = tf.layers.conv2d(layers4_2, filters = 96, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_3')#(30, 35, 35, 96)

    #---------------------------------------------------------------------------------
    layers5 = tf.concat([layers1,layers2,layers3_2,layers4_3], axis = -1)

    return layers5


def inception_b(name, inputs):
  with tf.variable_scope(name):
    #----------------------------------------------------------------------------------
    #line1
    pool1 = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME', name = name +'_pool1')

    layers1 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers1')

    #--------------------------------------------------------------------------------
    #line2
    layers2 = tf.layers.conv2d(inputs, filters = 384, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2')

    #--------------------------------------------------------------------------------
    #line3
    layers3_1 = tf.layers.conv2d(inputs, filters = 192, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_1')

    layers3_2 = tf.layers.conv2d(layers3_1, filters = 224, kernel_size =[1, 7], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_2')

    layers3_3 = tf.layers.conv2d(layers3_2, filters = 256, kernel_size =[1, 7], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_3')

    #------------------------------------------------------------------------------
    #line4
    layers4_1 = tf.layers.conv2d(inputs, filters = 192, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_1')

    layers4_2 = tf.layers.conv2d(layers4_1, filters = 192, kernel_size =[1, 7], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_2')

    layers4_3 = tf.layers.conv2d(layers4_2, filters = 224, kernel_size =[7, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_3')

    layers4_4 = tf.layers.conv2d(layers4_3, filters = 224, kernel_size =[1, 7], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_4')

    layers4_5 = tf.layers.conv2d(layers4_4, filters = 256, kernel_size =[7, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_5')

    #----------------------------------------------------------------------------------
    #concat
    layers5 = tf.concat([layers1, layers2, layers3_3, layers4_5], axis = -1)

    return layers5



def inception_c(name, inputs):
  with tf.variable_scope(name):
    #----------------------------------------------------------------------------------
    #line1
    pool1 = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME', name = name +'_pool1')

    layers1 = tf.layers.conv2d(pool1, filters = 256, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers1')

    #----------------------------------------------------------------------------------
    #line2
    layers2 = tf.layers.conv2d(inputs, filters = 256, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2')

    #-----------------------------------------------------------------------------------
    #line3
    layers3_1 = tf.layers.conv2d(inputs, filters = 384, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_1')

    layers3_2 = tf.layers.conv2d(layers3_1, filters = 256, kernel_size = [1, 3], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_2')

    layers3_2_1 = tf.layers.conv2d(layers3_1, filters = 256, kernel_size = [3, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers3_1_1')

    #----------------------------------------------------------------------------------
    #line4
    layers4_1 = tf.layers.conv2d(inputs, filters = 384, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_1')

    layers4_2 = tf.layers.conv2d(layers4_1, filters = 448, kernel_size =[1, 3], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_2')

    layers4_3 = tf.layers.conv2d(layers4_2, filters = 512, kernel_size =[3, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_3')

    layers4_4 = tf.layers.conv2d(layers4_3, filters = 256, kernel_size =[3, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_4')

    layers4_4_1 = tf.layers.conv2d(layers4_3, filters = 256, kernel_size =[1, 3], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers4_4_1')

    #---------------------------------------------------------------------------------
    #concat
    layers5 = tf.concat([layers1, layers2, layers3_2, layers3_2_1, layers4_4, layers4_4_1], axis = -1)

    return layers5


def reduction_a(name, inputs):

  with tf.variable_scope(name):
    #-----------------------------------------------------
    #line1
    pool1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID', name = name +'_pool1')

    #-------------------------------------------------------
    #line2
    layers1 = tf.layers.conv2d(inputs, filters = 384, kernel_size = 3, strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers1')

    #------------------------------------------------------
    #line3
    layers2_1 = tf.layers.conv2d(inputs, filters = 192, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2_1')

    layers2_2 = tf.layers.conv2d(layers2_1, filters = 224, kernel_size = 3, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2_2')

    layers2_3 = tf.layers.conv2d(layers2_2, filters = 256, kernel_size = 3, strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers2_3')

    #-------------------------------------------
    #concat

    layers3 = tf.concat([pool1, layers1, layers2_3], axis = -1)

    return layers3

def reduction_b(name, inputs):
  with tf.variable_scope(name):
    #-----------------------------------------------------
    #line1
    pool1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID', name = name +'_pool1')

    #-----------------------------------------------------
    #line2
    layers1_1 = tf.layers.conv2d(inputs, filters = 192, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers1_1')

    layers1_2 = tf.layers.conv2d(layers1_1, filters = 192, kernel_size = 3, strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers1_2')

    #-----------------------------------------------------
    #line3
    layers2_1 = tf.layers.conv2d(inputs, filters = 256, kernel_size = 1, strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2_1')

    layers2_2 = tf.layers.conv2d(layers2_1, filters = 256, kernel_size = [1, 7], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2_2')

    layers2_3 = tf.layers.conv2d(layers2_2, filters = 320, kernel_size = [7, 1], strides = 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'same', name = name +'_layers2_3')

    layers2_4 = tf.layers.conv2d(layers2_3, filters = 320, kernel_size = [3, 3], strides = 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu,
                     padding = 'valid', name = name +'_layers2_4')

    #-----------------------------------------------------
    #concat
    layers3 = tf.concat([pool1, layers1_2, layers2_4], axis = -1)

    return layers3



def model(inputs):
  #inputs(batch,299,299,3)
  net = stem('stem', inputs)
  #-------------------------------------
  net = inception_a('inception_a1', net)#(30, 35, 35, 384)
  net = inception_a('inception_a2', net)
  net = inception_a('inception_a3', net)
  net = inception_a('inception_a4', net)#(30, 35, 35, 384)
  #-------------------------------------
  net = reduction_a('reduction_a', net)#(30, 17, 17, 1024)
  #-------------------------------------
  net = inception_b('inception_b1', net)
  net = inception_b('inception_b2', net)
  net = inception_b('inception_b3', net)
  net = inception_b('inception_b4', net)
  net = inception_b('inception_b5', net)
  net = inception_b('inception_b6', net)
  net = inception_b('inception_b7', net)#(30, 17, 17, 1024)
  #-------------------------------------
  net = reduction_b('reduction_b', net)#(30, 8, 8, 1536)
  #-------------------------------------
  net = inception_c('inception_c1', net)
  net = inception_c('inception_c2', net)#(30, 8, 8, 1536)
  net = inception_c('inception_c3', net)
  #-------------------------------------
  net = tf.nn.avg_pool(net, ksize=[1, 8, 8, 1],
                        strides=[1, 8, 8, 1], padding='VALID', name = 'pool')

  net = tf.squeeze(net)
  net = tf.nn.dropout(net, keep_prob = 0.8, name = 'drop_out')

  logits = tf.layers.dense(net, 30)#(30,30)

  return logits



