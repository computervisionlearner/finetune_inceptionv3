#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:29:36 2017

@author: no1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime

import random

import sys
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
import glob

FLAGS = None


BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, validation_percentage):
  train_data = {}
  valid_data = {}
  for root, subs, files in os.walk(image_dir):
    for sub in subs:
      new_path = os.path.join(root,sub)
      path = glob.glob(new_path+'/*.jpg')
      random.shuffle(path)
      train_data[sub] = path[int(validation_percentage * len(path)):]
      valid_data[sub] = path[:int(validation_percentage * len(path))]
      
  return train_data, valid_data

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    图， 倒数第二层输出， 照片输入，resize后的照片输入
  """
  with tf.Graph().as_default() as graph:
    model_filename = os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME])
  
  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):

  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def create_bottleneck_file(bottleneck_path, img_path, sess, jpeg_data_tensor, bottleneck_tensor):
  
  """Create a single bottleneck file."""
  
  print('\r>> Creating bottleneck at {}'.format(bottleneck_path))
  if not gfile.Exists(img_path):
    tf.logging.fatal('File does not exist %s', img_path)
  image_data = gfile.FastGFile(img_path, 'rb').read()
  try:
    one_img_bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, bottleneck_tensor)
  except:
    raise RuntimeError('Error during processing file %s' % img_path)

  bottleneck_string = ','.join(str(x) for x in one_img_bottleneck_values)
  
  with open(bottleneck_path, 'w') as bottleneck_file:
    
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, img_path, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):

  #/home/sw/Documents/inception_v3_fs/data/1/1_123.jpg
  sub_dir = img_path.split('/')[-2]
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottlefile_name = os.path.basename(img_path).split('.')[0] + '.txt'
  bottleneck_path = os.path.join(sub_dir_path, bottlefile_name)

  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, img_path, sess, jpeg_data_tensor, bottleneck_tensor)

  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, img_path, sess, jpeg_data_tensor, bottleneck_tensor)
    
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()

    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, train_lists, valid_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):

  count = 0
  ensure_dir_exists(bottleneck_dir)#如果文件夹不存在，就新建
  
  all_imgpath =list(train_lists.values()) + list(valid_lists.values())
  all_imgs = list(np.concatenate(all_imgpath))
  for img_path in all_imgs:
    get_or_create_bottleneck(sess, img_path, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
    count += 1
    sys.stdout.write('\r>>{}/{} bottleneck files created'.format(count, len(all_imgs)))
    sys.stdout.flush()


def get_random_cached_bottlenecks(sess, image_lists, train_batch_size, bottleneck_dir, image_dir, jpeg_data_tensor,bottleneck_tensor):
  #how_many is batch_size
  class_count = len(image_lists.keys())
  bottlenecks = []
  labels = []

  # Retrieve a random sample of bottlenecks.
  for unused_i in range(train_batch_size):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(len(image_lists[label_name]))
    img_path = image_lists[label_name][image_index]
    
    bottleneck = get_or_create_bottleneck(sess, img_path, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
    
    label = np.zeros(class_count, dtype=np.float32)
    label[label_index] = 1.0
    bottlenecks.append(bottleneck)
    labels.append(label)

  return bottlenecks, labels


def get_random_distorted_bottlenecks(
    sess, train_lists, train_batch_size,
             image_dir, jpeg_string_placeholder,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor):

  class_count = len(train_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(train_batch_size):
    label_index = random.randrange(class_count)
    label_name = list(train_lists.keys())[label_index]
    image_index = random.randrange(len(train_lists[label_name]))
    image_path = train_lists[label_name][image_index]
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()

    distorted_image_data = sess.run(distorted_image_tensor,
                                    {jpeg_string_placeholder: jpeg_data})
    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                         resized_image_tensor,
                                         bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):

  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):

  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
 
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

def preprocess_for_train(img, height, width, bbox,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
 
  with tf.name_scope(scope, 'distort_image', [img, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    if img.dtype != tf.float32:
      img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    distorted_image1, distorted_bbox1 = distorted_bounding_box_crop(img, bbox)

    distorted_image1.set_shape([None, None, 3])

    distorted_image2 = tf.image.resize_images(distorted_image1, size = (height, width), method = random.randint(0, 3))

    distorted_image3 = tf.image.random_flip_left_right(distorted_image2)
    # Randomly distort the colors. There are 4 ways to do it.
#    distorted_image4 = distort_color(distorted_image3, color_ordering = random.randint(0,2), fast_mode = fast_mode)
        
#    distorted_image5 = tf.subtract(distorted_image4, 0.5)
#    distorted_image6 = tf.multiply(distorted_image5, 2.0)
    distorted_image3 =tf.expand_dims( tf.image.convert_image_dtype(distorted_image3, dtype = tf.uint8),0)
    return distorted_image3
  
def add_input_distortions():

  '''
    只有选择预处理，才运行改函数
  '''
  jpeg_placeholder = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_placeholder, channels=MODEL_INPUT_DEPTH)
  
  distort_result = preprocess_for_train(decoded_image, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, bbox = None)
  
  return jpeg_placeholder, distort_result


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):

  with tf.name_scope('input'):
    bottleneck_input_placeholder = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    labels_placeholder = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                                          stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input_placeholder, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_placeholder, logits=logits)
    with tf.name_scope('total'):
      loss = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', loss)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(loss)

  return train_op, loss, bottleneck_input_placeholder, labels_placeholder, final_tensor


def add_evaluation_step(result_tensor, ground_truth_tensor):

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def main(_):
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Set up the pre-trained graph.
  #  图， 倒数第二层输出， 照片输入，resize后的照片输入
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()
  train_lists, valid_lists = create_image_lists(FLAGS.image_dir, FLAGS.validation_percentage)

  with tf.Session(graph=graph) as sess:

    if FLAGS.preprocess:
      # We will be applying distortions, so setup the operations we'll need.
      #返回值： 二进制占位符，预处理之后的值
      jpeg_string_placeholder, distorted_image_tensor = add_input_distortions()
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, train_lists, valid_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        bottleneck_tensor)

    # Add the new layer that we'll be training.
    #训练节点，交叉熵，两个占位符，最后的输出（softmax的输出）
    train_op, loss, bottleneck_input_placeholder, labels_placeholder, final_tensor = add_final_training_ops(len(train_lists.keys()),FLAGS.final_tensor_name,
                                            bottleneck_tensor)
    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(final_tensor, labels_placeholder)
    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)
    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.how_many_training_steps):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if FLAGS.preprocess:

        batch_bottlenecks, batch_labels = get_random_distorted_bottlenecks(
             sess, train_lists, FLAGS.train_batch_size,
             FLAGS.image_dir, jpeg_string_placeholder,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        batch_bottlenecks, batch_labels = get_random_cached_bottlenecks(sess, train_lists, FLAGS.train_batch_size, FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,bottleneck_tensor)

      train_summary, _ = sess.run([merged, train_op], feed_dict={bottleneck_input_placeholder: batch_bottlenecks,
                     labels_placeholder: batch_labels})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, loss],
            feed_dict={bottleneck_input_placeholder: batch_bottlenecks,
                       labels_placeholder: batch_labels})
  
        output_graph_def = graph_util.convert_variables_to_constants(
          sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
        
        with gfile.FastGFile(FLAGS.output_graph+'{}'.format(i), 'wb') as f:
          f.write(output_graph_def.SerializeToString())
          
        with gfile.FastGFile(FLAGS.output_labels+'{}'.format(i), 'w') as f:
          f.write('\n'.join(train_lists.keys()) + '\n')
        print('%s: Step %d: Train accuracy = %.3f%%' % (datetime.now(), i,
                                                        train_accuracy * 100))
        print('%s: Step %d: Cross entropy = %.3f' % (datetime.now(), i,
                                                   cross_entropy_value))
        validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, valid_lists, FLAGS.train_batch_size, FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,bottleneck_tensor)
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step], feed_dict={bottleneck_input_placeholder: validation_bottlenecks,
                       labels_placeholder: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        print('%s: Step %d: Validation accuracy = %.3f%% (N=%d)' %
              (datetime.now(), i, validation_accuracy * 100,
               len(validation_bottlenecks)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='data',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=40000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )

  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=0.10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=100,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )

  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )

  parser.add_argument(
      '--model_dir',
      type=str,
      default='imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--preprocess',
      default=True,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
