# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy
import numpy as np
import tensorflow as tf
import pdb
import random
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
slim = tf.contrib.slim
from scipy.misc import imresize

def parse_tf_example(serialized):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized TF Example.
  Returns:
    encoded_image: A scalar string Tensor containing an encoded image.
    seg_mask: A h X w [0,1] Tensor indicating the tampered object.
    edge_mask: A h X w [0,1] Tensor indicating tampered edge.
  """
  features = tf.parse_single_example(
      serialized,
      features={
          
          "shape": tf.FixedLenFeature([], tf.string),
          "label": tf.FixedLenFeature([], tf.int64),
          "name": tf.FixedLenFeature([], tf.string),
          #"width": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string),
          #"edge": tf.FixedLenFeature([], tf.string),
          "mask": tf.FixedLenFeature([], tf.string),
          
      }
  )

  encoded_image = features["image"]
  seg_mask = features["mask"]
  #edge_mask = features["edge"]

  #height = tf.cast(features["shape"], tf.int32)
  #width = tf.cast(features["width"], tf.int32)

  #return (encoded_image, seg_mask, edge_mask,
          #features["image_name"], features["label"])
  return (encoded_image, seg_mask,
          features["name"], features["label"])

def parse_tf_example_pair(serialized):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized TF Example.
  Returns:
    encoded_image: A scalar string Tensor containing an encoded image.
    seg_mask: A h X w [0,1] Tensor indicating the tampered object.
    edge_mask: A h X w [0,1] Tensor indicating tampered edge.
  """
  features = tf.parse_single_example(
      serialized,
      features={
          
          "shape": tf.FixedLenFeature([], tf.string),
          "label": tf.FixedLenFeature([], tf.int64),
          "name": tf.FixedLenFeature([], tf.string),
          #"width": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string),
          "tamper": tf.FixedLenFeature([], tf.string),
          "mask": tf.FixedLenFeature([], tf.string),
          
      }
  )

  encoded_image = features["image"]
  seg_mask = features["mask"]
  tamper_im = features['tamper']
  #edge_mask = features["edge"]

  #height = tf.cast(features["shape"], tf.int32)
  #width = tf.cast(features["width"], tf.int32)

  #return (encoded_image, seg_mask, edge_mask,
          #features["image_name"], features["label"])
  return (encoded_image, tamper_im, seg_mask,
          features["name"], features["label"])

def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)
  #pdb.set_trace()
  if is_training:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=2000, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    #pdb.set_trace()
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def distort_image(image, thread_id):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  #color_ordering = thread_id % 2
  # with tf.name_scope("distort_color", values=[image]):
  #  if color_ordering == 0:
  #    image = tf.image.random_brightness(image, max_delta=32. / 255.)
  #    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
  #    image = tf.image.random_hue(image, max_delta=0.032)
  #    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
  #  elif color_ordering == 1:
  #    image = tf.image.random_brightness(image, max_delta=32. / 255.)
  #    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
  #    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
  #    image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
  #  image = tf.clip_by_value(image, 0.0, 1.0)

  return image

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.2 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.2, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
   
    return img, label

def image_mirroring(img, label,gt_img):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    gt_img = tf.reverse(gt_img, mirror)
    label = tf.reverse(label, mirror)
    return img, label, gt_img

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def process_image(encoded_image,
                  seg_mask,
                  #edge_mask,
                  is_training,
                  random_mirror=False,
                  random_scale=False,
                  image_mean=np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32),
                  is_authentic=True,
                  height=256,
                  width=256,
                  resize_height=256,
                  resize_width=256,
                  thread_id=0,
                  image_format="raw"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    seg_mask: Matrix containing the segmentation of tampreed rigion.
    edge_mask: Matrix containing the segmentation of tampered edge.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".
  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
      seg_mask = tf.image.decode_jpeg(seg_mask, channels=1)
      #edge_mask = tf.image.decode_jpeg(edge_mask, channels=1)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
      seg_mask = tf.image.decode_png(seg_mask, channels=1)
      #edge_mask = tf.image.decode_png(edge_mask, channels=1)
    elif image_format == "raw":
      image = tf.decode_raw(encoded_image, tf.uint8)
      #pdb.set_trace()
      image=tf.reshape(image,[height,width,3])
      seg_mask = tf.decode_raw(seg_mask, tf.uint8)
      seg_mask = tf.reshape(seg_mask,[height,width,1])
    else:
      raise ValueError("Invalid image format: %s" % image_format)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  #image = tf.cast(image, dtype=tf.float32)
  seg_mask = tf.image.convert_image_dtype(seg_mask, dtype=tf.float32)
  #seg_mask = tf.cast(seg_mask, dtype=tf.float32)
  #edge_mask = tf.cast(edge_mask, dtype=tf.float32)

  #image_summary("original_image", image[:,:,::-1])
  #image_summary("original_image", image[:,:,0:3])
  #image_summary("original_seg_mask", seg_mask)
  #image_summary("original_edge_mask", edge_mask)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  seg_mask = tf.image.resize_images(seg_mask,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #edge_mask = tf.image.resize_images(edge_mask,
                                #size=[resize_height, resize_width],
                                #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # Crop to final dimensions.
  #joint_input = tf.concat([image, seg_mask, edge_mask], axis=-1)

  
  if is_training:
    #if random_scale:
        #image, seg_mask = image_scaling(image, seg_mask)

    # Randomly mirror the images and labels.
    #if random_mirror:
        #image, seg_mask = image_mirroring(image, seg_mask)  
    joint_input = tf.concat([image, seg_mask], axis=-1)  
    #joint_input = tf.random_crop(joint_input, [height, width, 5])
    joint_input = tf.random_crop(joint_input, [resize_height, resize_height, 4])
  else:
    joint_input = tf.concat([image, seg_mask], axis=-1)
    # Central crop, assuming resize_height > height, resize_width > width.
    joint_input = tf.image.resize_image_with_crop_or_pad(joint_input, resize_height, resize_height)

  # Randomly distort the image.
  if is_training:
    joint_input = distort_image(joint_input, thread_id)

  image = joint_input[:,:,:3]
  seg_mask = joint_input[:,:,3:4]

  tamper_region = tf.image.random_brightness(image,max_delta=64. / 255.)
  image_summary("brightness", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_saturation(tamper_region, lower=0.3, upper=1.7)
  image_summary("saturation", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_hue(tamper_region, max_delta=0.1)
  image_summary("hue", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_contrast(tamper_region, lower=0.3, upper=1.7)
  image_summary("contrast", tamper_region[:,:,::-1])
  #tamper_region = gaussian_noise_layer(tamper_region, 0.1)
  #image_summary("add noise", tamper_region[:,:,::-1])
  tamper_region = tf.clip_by_value(tamper_region, 0.0, 1.0)

  if random_mirror:
  #if False:
    distort=tf.concat([tamper_region, seg_mask,image], axis=-1)
    distort = tf.image.random_flip_left_right(distort,seed=3)
    tamper_region=distort[:,:,:3]
    seg_mask1=distort[:,:,3:4]
    image_m=distort[:,:,4:]
    #tamper_region, seg_mask = image_scaling(tamper_region, seg_mask)
    #tamper_scale=tf.concat([tamper_region, seg_mask], axis=-1)
    #tamper_scale = tf.image.resize_image_with_crop_or_pad(tamper_scale, resize_height, resize_height)
    #tamper_region=tamper_scale[:,:,:3]
    #seg_mask = tamper_scale[:,:,3:4]
    #tamper_region, seg_mask, image_m = image_mirroring(tamper_region,seg_mask,image)
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask1,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask1,tf.float32)) + tamper_region  
    if not is_authentic:
      image_summary("final_tamper", image_distort[:,:,::-1])
      #edge_mask = joint_input[:,:,4:]
      
      # print(image)
      # print(seg_mask)
      # print(edge_mask)
      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      image_distort = image_distort*255-image_mean
      image = image*255-image_mean
      #image = image - image_mean
      #return image, seg_mask, edge_mask
      seg_mask = tf.maximum(seg_mask, seg_mask1)
      return image, seg_mask, image_distort      
    else:
      image_summary("final_tamper", image_distort[:,:,::-1])
      #edge_mask = joint_input[:,:,4:]
      
      # print(image)
      # print(seg_mask)
      # print(edge_mask)
      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      image_distort = image_distort*255-image_mean
      image = image*255-image_mean
      #image = image - image_mean
      #return image, seg_mask, edge_mask
      return image, seg_mask1, image_distort
    #image = image_m  
  else:
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask,tf.float32)) + tamper_region
    image_summary("final_tamper", image_distort[:,:,::-1])
    #edge_mask = joint_input[:,:,4:]
    
    # print(image)
    # print(seg_mask)
    # print(edge_mask)
    image_summary("final_image", image[:,:,::-1])
    #image_summary("final_image", image[:,:,::-1])
    image_summary("final_seg_mask", seg_mask)
    #image_summary("final_edge_mask", edge_mask)

    # Rescale to image-mean instead of [0, 1]
    image_distort = image_distort*255-image_mean
    image = image*255-image_mean
    #image = image - image_mean
    #return image, seg_mask, edge_mask
    return image, seg_mask, image_distort

def process_image_unet(encoded_image,
                  seg_mask,
                  #edge_mask,
                  is_training,
                  random_mirror=False,
                  random_scale=False,
                  image_mean=np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32),
                  is_authentic=True,
                  height=256,
                  width=256,
                  resize_height=256,
                  resize_width=256,
                  thread_id=0,
                  image_format="raw"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    seg_mask: Matrix containing the segmentation of tampreed rigion.
    edge_mask: Matrix containing the segmentation of tampered edge.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".
  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
      seg_mask = tf.image.decode_jpeg(seg_mask, channels=1)
      #edge_mask = tf.image.decode_jpeg(edge_mask, channels=1)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
      seg_mask = tf.image.decode_png(seg_mask, channels=1)
      #edge_mask = tf.image.decode_png(edge_mask, channels=1)
    elif image_format == "raw":
      image = tf.decode_raw(encoded_image, tf.uint8)
      #pdb.set_trace()
      image=tf.reshape(image,[height,width,3])
      seg_mask = tf.decode_raw(seg_mask, tf.uint8)
      seg_mask = tf.reshape(seg_mask,[height,width,1])
    else:
      raise ValueError("Invalid image format: %s" % image_format)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  #image = tf.cast(image, dtype=tf.float32)
  seg_mask = tf.image.convert_image_dtype(seg_mask, dtype=tf.float32)
  #seg_mask = tf.cast(seg_mask, dtype=tf.float32)
  #edge_mask = tf.cast(edge_mask, dtype=tf.float32)

  #image_summary("original_image", image[:,:,::-1])
  #image_summary("original_image", image[:,:,0:3])
  #image_summary("original_seg_mask", seg_mask)
  #image_summary("original_edge_mask", edge_mask)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  seg_mask = tf.image.resize_images(seg_mask,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #edge_mask = tf.image.resize_images(edge_mask,
                                #size=[resize_height, resize_width],
                                #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # Crop to final dimensions.
  #joint_input = tf.concat([image, seg_mask, edge_mask], axis=-1)

  
  if is_training:
    #if random_scale:
        #image, seg_mask = image_scaling(image, seg_mask)

    # Randomly mirror the images and labels.
    #if random_mirror:
        #image, seg_mask = image_mirroring(image, seg_mask)  
    joint_input = tf.concat([image, seg_mask], axis=-1)  
    #joint_input = tf.random_crop(joint_input, [height, width, 5])
    joint_input = tf.random_crop(joint_input, [resize_height, resize_height, 4])
  else:
    joint_input = tf.concat([image, seg_mask], axis=-1)
    # Central crop, assuming resize_height > height, resize_width > width.
    joint_input = tf.image.resize_image_with_crop_or_pad(joint_input, resize_height, resize_height)

  # Randomly distort the image.
  if is_training:
    joint_input = distort_image(joint_input, thread_id)

  image = joint_input[:,:,:3]
  seg_mask = joint_input[:,:,3:4]

  tamper_region = tf.image.random_brightness(image,max_delta=32. / 255.)
  image_summary("brightness", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_saturation(tamper_region, lower=0.5, upper=1.5)
  image_summary("saturation", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_hue(tamper_region, max_delta=0.032)
  image_summary("hue", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_contrast(tamper_region, lower=0.5, upper=1.5)
  image_summary("contrast", tamper_region[:,:,::-1])
  #tamper_region = gaussian_noise_layer(tamper_region, 0.1)
  #image_summary("add noise", tamper_region[:,:,::-1])
  tamper_region = tf.clip_by_value(tamper_region, 0.0, 1.0)

  if random_mirror:
  #if False:
    distort=tf.concat([tamper_region, seg_mask,image], axis=-1)
    distort = tf.image.random_flip_left_right(distort,seed=3)
    tamper_region=distort[:,:,:3]
    seg_mask1=distort[:,:,3:4]
    image_m=distort[:,:,4:]
    #tamper_region, seg_mask = image_scaling(tamper_region, seg_mask)
    #tamper_scale=tf.concat([tamper_region, seg_mask], axis=-1)
    #tamper_scale = tf.image.resize_image_with_crop_or_pad(tamper_scale, resize_height, resize_height)
    #tamper_region=tamper_scale[:,:,:3]
    #seg_mask = tamper_scale[:,:,3:4]
    #tamper_region, seg_mask, image_m = image_mirroring(tamper_region,seg_mask,image)
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask1,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask1,tf.float32)) + tamper_region  
    if not is_authentic:
      image_summary("final_tamper", image_distort[:,:,::-1])
      #edge_mask = joint_input[:,:,4:]
      
      # print(image)
      # print(seg_mask)
      # print(edge_mask)
      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      #image_distort = image_distort*255-image_mean
      #image = image*255-image_mean
      #image = image - image_mean
      #return image, seg_mask, edge_mask
      seg_mask = tf.maximum(seg_mask, seg_mask1)
      return image, seg_mask, image_distort      
    else:
      image_summary("final_tamper", image_distort[:,:,::-1])
      #edge_mask = joint_input[:,:,4:]
      
      # print(image)
      # print(seg_mask)
      # print(edge_mask)
      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      #image_distort = image_distort*255-image_mean
      #image = image*255-image_mean
      #image = image - image_mean
      #return image, seg_mask, edge_mask
      return image, seg_mask1, image_distort
    #image = image_m  
  else:
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask,tf.float32)) + tamper_region
    image_summary("final_tamper", image_distort[:,:,::-1])
    #edge_mask = joint_input[:,:,4:]
    
    # print(image)
    # print(seg_mask)
    # print(edge_mask)
    image_summary("final_image", image[:,:,::-1])
    #image_summary("final_image", image[:,:,::-1])
    image_summary("final_seg_mask", seg_mask)
    #image_summary("final_edge_mask", edge_mask)

    # Rescale to image-mean instead of [0, 1]
    image_distort = image_distort*255-image_mean
    image = image*255-image_mean
    #image = image - image_mean
    #return image, seg_mask, edge_mask
    return image, seg_mask, image_distort

def process_image_pair(encoded_image,
                  tamper_im,
                  seg_mask,                  
                  is_training,
                  random_mirror=False,
                  random_scale=False,
                  image_mean=np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32),
                  is_authentic=True,
                  height=256,
                  width=256,
                  resize_height=256,
                  resize_width=256,
                  thread_id=0,
                  image_format="raw"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    seg_mask: Matrix containing the segmentation of tampreed rigion.
    edge_mask: Matrix containing the segmentation of tampered edge.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".
  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
      seg_mask = tf.image.decode_jpeg(seg_mask, channels=1)
      #edge_mask = tf.image.decode_jpeg(edge_mask, channels=1)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
      seg_mask = tf.image.decode_png(seg_mask, channels=1)
      #edge_mask = tf.image.decode_png(edge_mask, channels=1)
    elif image_format == "raw":
      image = tf.decode_raw(encoded_image, tf.uint8)
      tamper_image = tf.decode_raw(tamper_im, tf.uint8)
      
      tamper_image=tf.reshape(tamper_image,[height,width,3])
      image=tf.reshape(image,[height,width,3])
      seg_mask = tf.decode_raw(seg_mask, tf.uint8)
      seg_mask = tf.reshape(seg_mask,[height,width,1])
    else:
      raise ValueError("Invalid image format: %s" % image_format)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  tamper_image = tf.image.convert_image_dtype(tamper_image, dtype=tf.float32)
  #image = tf.cast(image, dtype=tf.float32)
  seg_mask = tf.image.convert_image_dtype(seg_mask, dtype=tf.float32)
  #seg_mask = tf.cast(seg_mask, dtype=tf.float32)
  #edge_mask = tf.cast(edge_mask, dtype=tf.float32)

  #image_summary("original_image", image[:,:,::-1])
  #image_summary("original_image", image[:,:,0:3])
  #image_summary("original_seg_mask", seg_mask)
  #image_summary("original_edge_mask", edge_mask)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  tamper_image = tf.image.resize_images(tamper_image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  seg_mask = tf.image.resize_images(seg_mask,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  if False:
    kernel_tensor = tf.constant(np.ones((5,5,1),np.float32))
    seg_mask_new=tf.squeeze(tf.nn.erosion2d(tf.expand_dims(seg_mask, dim=0),kernel_tensor,
                               strides=[1,1,1,1],
                               rates=[1,1,1,1],
                               padding='SAME')+1.0,axis=0)
    tamper_image_new=tf.multiply(tamper_image, seg_mask_new) + tf.multiply(image, 1-seg_mask_new)
    joint0=tf.concat([image, tamper_image, seg_mask], axis=-1)  
    joint1=tf.concat([image, tamper_image_new, seg_mask_new], axis=-1)
    random_int = tf.random_uniform([1],minval=0, maxval=100,dtype=tf.int32,seed=1234)
    def f0(): return joint0
    def f1(): return joint1
    joint = tf.case({tf.equal(random_int[0]%2,0):f0,
                            tf.equal(random_int[0]%2,1):f1}, default=f1, exclusive=True)
    image = joint[:,:,:3]
    tamper_image= joint[:,:,3:6]
    seg_mask = joint[:,:,6:7]
  #edge_mask = tf.image.resize_images(edge_mask,
                                #size=[resize_height, resize_width],
                                #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # Crop to final dimensions.
  #joint_input = tf.concat([image, seg_mask, edge_mask], axis=-1)

  
  if is_training:
    #if random_scale:
        #image, seg_mask = image_scaling(image, seg_mask)

    # Randomly mirror the images and labels.
    #if random_mirror:
        #image, seg_mask = image_mirroring(image, seg_mask)  
    joint_input = tf.concat([image, tamper_image, seg_mask], axis=-1)  
    #joint_input = tf.random_crop(joint_input, [height, width, 5])
    joint_input = tf.random_crop(joint_input, [resize_height, resize_height, 7])
    #joint_input = tf.image.random_flip_left_right(joint_input,seed=3)
  else:
    joint_input = tf.concat([image,tamper_image, seg_mask], axis=-1)
    # Central crop, assuming resize_height > height, resize_width > width.
    joint_input = tf.image.resize_image_with_crop_or_pad(joint_input, resize_height, resize_height)


    #distort=tf.concat([tamper_region, seg_mask,tamper_image], axis=-1)
  # Randomly distort the image.
  if is_training:
    joint_input = distort_image(joint_input, thread_id)

  image = joint_input[:,:,:3]
  tamper_image= joint_input[:,:,3:6]
  seg_mask = joint_input[:,:,6:7]

  tamper_region = tf.image.random_brightness(tamper_image,max_delta=32. / 255.)
  image_summary("brightness", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_saturation(tamper_region, lower=0.5, upper=1.5)
  image_summary("saturation", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_hue(tamper_region, max_delta=0.032)
  image_summary("hue", tamper_region[:,:,::-1])
  tamper_region = tf.image.random_contrast(tamper_region, lower=0.5, upper=1.5)
  image_summary("contrast", tamper_region[:,:,::-1])
  #tamper_region = gaussian_noise_layer(tamper_region, 0.1)
  #image_summary("add noise", tamper_region[:,:,::-1])
  tamper_region = tf.clip_by_value(tamper_region, 0.0, 1.0)

  if random_mirror:
  #if False:
    distort=tf.concat([tamper_region, seg_mask,tamper_image], axis=-1)
    distort = tf.image.random_flip_left_right(distort,seed=3)
    tamper_region=distort[:,:,:3]
    seg_mask1=distort[:,:,3:4]
    image_m=distort[:,:,4:]
    #tamper_region, seg_mask = image_scaling(tamper_region, seg_mask)
    #tamper_scale=tf.concat([tamper_region, seg_mask], axis=-1)
    #tamper_scale = tf.image.resize_image_with_crop_or_pad(tamper_scale, resize_height, resize_height)
    #tamper_region=tamper_scale[:,:,:3]
    #seg_mask = tamper_scale[:,:,3:4]
    #tamper_region, seg_mask, image_m = image_mirroring(tamper_region,seg_mask,image)
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask1,tf.float32))
    image_distort = tf.multiply(tamper_image,1-tf.cast(seg_mask1,tf.float32)) + tamper_region  
    if not is_authentic:
      image_summary("final_tamper", image_distort[:,:,::-1])
      #edge_mask = joint_input[:,:,4:]
      
      # print(image)
      # print(seg_mask)
      # print(edge_mask)
      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      image_distort = image_distort*255-image_mean
      image = image*255-image_mean
      #image = image - image_mean
      #return image, seg_mask, edge_mask
      seg_mask = tf.maximum(seg_mask, seg_mask1)
      return image, tamper_image, seg_mask, image_distort      
    else:
      image_summary("final_tamper", image_distort[:,:,::-1])
      #edge_mask = joint_input[:,:,4:]
      
      # print(image)
      # print(seg_mask)
      # print(edge_mask)
      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      image_distort = image_distort*255-image_mean
      image = image*255-image_mean
      #image = image - image_mean
      #return image, seg_mask, edge_mask
      return image, tamper_image, seg_mask1, image_distort  
  else:

    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask,tf.float32))
    image_distort = tf.multiply(tamper_image,1-tf.cast(seg_mask,tf.float32)) + tamper_region
    image_summary("final_tamper", image_distort[:,:,::-1])
    #edge_mask = joint_input[:,:,4:]
    
    # print(image)
    # print(seg_mask)
    # print(edge_mask)
    image_summary("final_image", image[:,:,::-1])
    image_summary("tamper_image", tamper_image[:,:,::-1])
    image_summary("final_seg_mask", seg_mask)
    #image_summary("final_edge_mask", edge_mask)



    # Rescale to image-mean instead of [0, 1]
    image_distort = image_distort*255-image_mean
    tamper_image = tamper_image*255-image_mean
    image = image*255-image_mean
    #image = image - image_mean
    #return image, seg_mask, edge_mask
    return image, tamper_image, seg_mask, image_distort

def conv(batch_input, out_channels, stride,kernel=4, name="conv",padding='VALID'):
  if padding=='VALID':
    with tf.variable_scope(name):
      in_channels = batch_input.get_shape()[3]
      filter = tf.get_variable("filter",
                               [kernel, kernel, in_channels, out_channels],
                               dtype=tf.float32,
                               initializer=tf.random_normal_initializer(0, 0.02))
      # [batch, in_height, in_width, in_channels]
      # [filter_width, filter_height, in_channels, out_channels]
      #   => [batch, out_height, out_width, out_channels]
      padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                            1, 1], [0, 0]], mode="CONSTANT")
      conv = tf.nn.conv2d(padded_input, filter, [
                          1, stride, stride, 1], padding=padding)
      return conv
  else:
    with tf.variable_scope(name):
      in_channels = batch_input.get_shape()[3]
      filter = tf.get_variable("filter",
                               [kernel, kernel, in_channels, out_channels],
                               dtype=tf.float32,
                               initializer=tf.random_normal_initializer(0, 0.02))
      # [batch, in_height, in_width, in_channels]
      # [filter_width, filter_height, in_channels, out_channels]
      #   => [batch, out_height, out_width, out_channels]

      conv = tf.nn.conv2d(batch_input, filter, [
                          1, stride, stride, 1], padding=padding)
      return conv

def final_conv(batch_input, out_channels=1, stride=1):
  with tf.variable_scope("conv"):
    in_channels = batch_input.get_shape()[3]
    filter = tf.get_variable("filter",
                         [4, 3, in_channels, out_channels],
                         dtype=tf.float32,
                         initializer=tf.random_normal_initializer(0, 0.02))

    conv = tf.nn.conv2d(batch_input, filter, [
                        1, stride, stride, 1], padding="VALID")
    return conv



# # always keep batchnorm in training mode
# def batchnorm(input):
#   with tf.variable_scope("batchnorm"):
#     # this block looks like it has 3 inputs on the graph unless we do this
#     input = tf.identity(input)

#     channels = input.get_shape()[3]
#     offset = tf.get_variable("offset",
#                              [channels],
#                              dtype=tf.float32,
#                              initializer=tf.zeros_initializer())
#     scale = tf.get_variable("scale",
#                       [channels],
#                       dtype=tf.float32,
#                       initializer=tf.random_normal_initializer(1.0, 0.02))
#     mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
#     variance_epsilon = 1e-5
#     normalized = tf.nn.batch_normalization(input, mean, variance, offset,
#                             scale, variance_epsilon=variance_epsilon)
#     return normalized

# seperate batch norm training and testing

def dice_loss(label, logits, smooth=1e-9):
  label = tf.cast(label, tf.float32)
  logit_1 = logits[:,1]
  logits_sum = tf.reduce_sum(logit_1)
  label_sum = tf.reduce_sum(label)
  dice_loss = 1- (2*tf.reduce_sum(logit_1 * label)+smooth)/(logits_sum+label_sum+smooth)
  return dice_loss

def batchnorm(inputs, is_training=True, decay=0.999, scope='batchnorm'):
  with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as scope:
    
    scale = tf.get_variable(name='scale',initializer=tf.ones([inputs.get_shape()[-1]]), trainable=True)
    beta = tf.get_variable( name='beta', initializer=tf.zeros([inputs.get_shape()[-1]]), trainable=True)
    #scale = tf.get_variable(tf.ones([inputs.get_shape()[-1]]), name='scale')
    #beta = tf.get_variable(tf.zeros([inputs.get_shape()[-1]]), name='beta')
    pop_mean = tf.get_variable(name='pop_mean', initializer=tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.get_variable(name='pop_var',initializer=tf.ones([inputs.get_shape()[-1]]), trainable=False)
    epsilon = 1e-5
    if is_training:
      batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
      # batch_mean, batch_var = tf.nn.moments(inputs, [0])
      train_mean = tf.assign(pop_mean,
                             pop_mean * decay + batch_mean * (1 - decay))
      train_var = tf.assign(pop_var,
                            pop_var * decay + batch_var * (1 - decay))
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs,
                                         batch_mean, batch_var, beta, scale, epsilon)
    else:
      return tf.nn.batch_normalization(inputs,
                                       pop_mean, pop_var, beta, scale, epsilon)

def batchnorm_old(inputs, is_training):
  #with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as scope:
    #pdb.set_trace()
  return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=is_training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def deconv(batch_input, out_channels):
  with tf.variable_scope("deconv"):
    batch, in_height, in_width, in_channels = [
        int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("filter",
                             [4, 4, out_channels, in_channels],
                             dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels]
    #  [filter_width, filter_height, out_channels, in_channels]
    #   => [batch, out_height, out_width, out_channels]
    conv = tf.nn.conv2d_transpose(batch_input,
                          filter,
                          [batch, in_height * 2, in_width * 2, out_channels],
                          [1, 2, 2, 1],
                          padding="SAME")
    return conv


# image and web summaries.
def save_images(fetches, image_dict, output_dir, step=None):
  # ["image", "product_image", "body_segment",
  # "prod_segment", "skin_segment", "outputs"]
  image_dir = os.path.join(output_dir, "images")
  if not os.path.exists(image_dir):
    os.makedirs(image_dir)

  filesets = []
  for i, in_path in enumerate(fetches["paths"]):
    if i >= 1:
      # continue
      break # only show up to 1 images for batch
    name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
    fileset = {"name": name, "step": step}
    for kind in image_dict:
      filename = name + "-" + kind + ".png"
      if step is not None:
        filename = "%08d-%s" % (step, filename)
      fileset[kind] = filename
      out_path = os.path.join(image_dir, filename)
      contents = fetches[kind][i]
      with open(out_path, "wb") as f:
        f.write(contents)
    filesets.append(fileset)
  return filesets


def append_index(filesets, image_dict, output_dir, step=False):
  # ["image", "product_image", "body_segment",
  # "prod_segment", "skin_segment", "outputs"]
  index_path = os.path.join(output_dir, "index.html")
  if os.path.exists(index_path):
    index = open(index_path, "a")
  else:
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")
    if step:
      index.write("<th>step</th>")
    index.write("<th>name</th><th>input</th>"
      "<th>output</th><th>target</th></tr>")

  for fileset in filesets:
    index.write("<tr>")

    if step:
      index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    for kind in image_dict:
      index.write("<td><img src='images/%s'></td>" % fileset[kind])

    index.write("</tr>")
  return index_path


# Inception V3 model

def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=False,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net





n_kernels = 3
kernel = np.zeros((5, 5, 3, n_kernels))
c = np.zeros((n_kernels, 5, 5))
c[0, 2, 1:4] = np.asarray([[1,-2,1]]) / 2.0
c[1, 1:4, 1:4] = np.asarray([[-1,2,-1], [2,-4,2], [-1,2,-1]]) / 4.0
c[2] = np.asarray([[-1,2,-2,2,-1],
                   [2,-6,8,-6,2],
                   [-2,8,-12,8,-2],
                   [2,-6,8,-6,2],
                   [-1,2,-2,2,-1]]) / 12.0

for i in range(n_kernels):
    kernel[:,:,0,i] = c[i,:,:]
    kernel[:,:,1,i] = c[i,:,:]
    kernel[:,:,2,i] = c[i,:,:]

def extract_noise_map(x):
  conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
  return conv
  # quantization
  # clip_conv = tf.clip_by_value(conv, -t, t)
  # return clip_conv




def lrelu(x, a=0.2):
  with tf.name_scope("lrelu"):
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


# # always keep batchnorm in training mode
# def batchnorm(input):
#   with tf.variable_scope("batchnorm"):
#     # this block looks like it has 3 inputs on the graph unless we do this
#     input = tf.identity(input)

#     channels = input.get_shape()[3]
#     offset = tf.get_variable("offset",
#                              [channels],
#                              dtype=tf.float32,
#                              initializer=tf.zeros_initializer())
#     scale = tf.get_variable("scale",
#                       [channels],
#                       dtype=tf.float32,
#                       initializer=tf.random_normal_initializer(1.0, 0.02))
#     mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
#     variance_epsilon = 1e-5
#     normalized = tf.nn.batch_normalization(input, mean, variance, offset,
#                             scale, variance_epsilon=variance_epsilon)
#     return normalized

# seperate batch norm training and testing

def deconv(batch_input, out_channels):
  with tf.variable_scope("deconv"):
    batch, in_height, in_width, in_channels = [
        int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("filter",
                             [4, 4, out_channels, in_channels],
                             dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels]
    #  [filter_width, filter_height, out_channels, in_channels]
    #   => [batch, out_height, out_width, out_channels]
    conv = tf.nn.conv2d_transpose(batch_input,
                          filter,
                          [batch, in_height * 2, in_width * 2, out_channels],
                          [1, 2, 2, 1],
                          padding="SAME")
    return conv


def focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-12
    #labels = tf.to_int64(labels)
    #labels = tf.convert_to_tensor(labels, tf.int64)
    #logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]
    logits = tf.nn.softmax(logits)
    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    #pdb.set_trace()
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    #pdb.set_trace()
    #reduced_fl = tf.reduce_sum(tf.reduce_max(fl,axis=1))
    reduced_fl = tf.reduce_mean(tf.reduce_max(fl,axis=1))
    #reduced_fl = tf.reduce_sum(fl)  # same as reduce_max
    return reduced_fl
def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    #pdb.set_trace()
    L=tf.reduce_mean(-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred+1.e-12)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred+1.e-12))
    return L
def balance_loss(labels, logits,batch_size):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    #labels = tf.to_int64(labels)
    #labels = tf.convert_to_tensor(labels, tf.int64)
    #logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]
    #pixel_num=logits.get_shape()[0]
    pixel_num = 256*256*batch_size
    logits = tf.nn.softmax(logits)
    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    neg = tf.squeeze(tf.where(tf.less(labels, 1)), 1)
    pos = tf.squeeze(tf.where(tf.equal(labels, 1)), 1)
    #pdb.set_trace()
    ce = tf.reduce_max(tf.multiply(onehot_labels, -tf.log(model_out)),axis=1)
    neg_loss=tf.reduce_mean((tf.reduce_sum(labels)/pixel_num)*tf.cast(tf.gather(ce, neg),tf.float64))
    pos_loss=tf.reduce_mean((1-tf.reduce_sum(labels)/pixel_num)*tf.cast(tf.gather(ce, pos),tf.float64))
    #ce_balance = tf.where(labels>0,1-tf.reduce_sum(labels)/pixel_num,tf.reduce_sum(labels)/pixel_num)*tf.multiply(onehot_labels, -tf.log(model_out))
    #weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    #fl = tf.multiply(alpha, tf.multiply(weight, ce))
    #pdb.set_trace()
    #reduced_fl = tf.reduce_sum(tf.reduce_max(fl,axis=1))
    reduced_fl = neg_loss+pos_loss
    #reduced_fl = tf.reduce_sum(fl)  # same as reduce_max
    return tf.cast(reduced_fl, tf.float32)
def CRF(image, corase_res,n_classes=2):
  import pydensecrf.densecrf as dcrf

  from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
      create_pairwise_gaussian, softmax_to_unary



  # The input should be the negative of the logarithm of probability values
  # Look up the definition of the softmax_to_unary for more information
  #pdb.set_trace()
  h, w, _ = corase_res.shape
  corase_res = corase_res.transpose(2, 0, 1).copy(order='C')
  U = -np.log(corase_res+1e-9).astype(np.float32)
  U = U.reshape((n_classes, -1))
  d = dcrf.DenseCRF2D(h, w, n_classes)
  #unary = softmax_to_unary(softmax)
  # The inputs should be C-continious -- we are using Cython wrapper

  #unary = np.ascontiguousarray(unary)
  
  #d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)
  d.setUnaryEnergy(U)

  # This potential penalizes small pieces of segmentation that are
  # spatially isolated -- enforces more spatially consistent segmentations
  feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

  d.addPairwiseEnergy(feats, compat=3,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

  # This creates the color-dependent features --
  # because the segmentation that we get from CNN are too coarse
  # and we can use local color features to refine them
  feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                     img=image, chdim=2)

  d.addPairwiseEnergy(feats, compat=10,
                       kernel=dcrf.DIAG_KERNEL,
                       normalization=dcrf.NORMALIZE_SYMMETRIC)
  Q = d.inference(5)

  res = np.array(Q, dtype=np.float32).reshape(
        (n_classes, h, w)).transpose(1, 2, 0)
  return res