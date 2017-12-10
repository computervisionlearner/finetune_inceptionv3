#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:26:10 2017

@author: no1
"""

import tensorflow as tf
import os
import numpy as np
FINAL_TENSORNAME = 'final_result:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
HEIGHT = 299
WIDTH = 299
lines = tf.gfile.GFile('output_labels.txt').readlines()
uid_to_human = {}
for uid,line in enumerate(lines) :
    line=line.strip('\n')
    uid_to_human[uid] = line
  
def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]

def preprocess_test(decoded_image, center_rate = 0.875, height = HEIGHT, width = WIDTH):
  with tf.name_scope('eval_image'):
    if decoded_image.dtype is not tf.float32:
      image = tf.image.convert_image_dtype(decoded_image, dtype = tf.float32)
      
    if center_rate:
      image = tf.image.central_crop(image, central_fraction= center_rate)
    if height and width:
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, size =(height, width), align_corners= False)
    image = tf.image.convert_image_dtype(image, dtype = tf.uint8)
    return image
  
  
graph = tf.Graph()

with tf.gfile.FastGFile('imagenet/classify_image_graph_def.pb','rb') as file:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(file.read())
  final_tensor, resized_input_tensor = tf.import_graph_def(graph_def, name='', return_elements=[FINAL_TENSORNAME,
              RESIZED_INPUT_TENSOR_NAME])

result = []
with tf.Session() as sess:
  for root, dirs, files in os.walk('images/'):
    for file in files:
      image_string = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
      decoded_image = tf.image.decode_jpeg(image_string, channels=3)
      processed_image = preprocess_test(decoded_image)
      predictions = sess.run(final_tensor, feed_dict = {resized_input_tensor: processed_image})
      predictions = np.squeeze(predictions)
      image_name = int(os.path.basename(file).split('.')[0])
      top_k = predictions.argsort()[::-1]
      for node_id in top_k:
        label_name = id_to_string[node_id]
        score = predictions[node_id]
        result.append([image_name, label_name, str('%.8f'%score)])
        
np.save('result',result)
        
    
  
  
  


