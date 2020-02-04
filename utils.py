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


  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  seg_mask = tf.image.resize_images(seg_mask,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


  # Crop to final dimensions.
  #joint_input = tf.concat([image, seg_mask, edge_mask], axis=-1)

  
  if is_training:

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

    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask1,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask1,tf.float32)) + tamper_region  
    if not is_authentic:
      image_summary("final_tamper", image_distort[:,:,::-1])

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


  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  image = tf.image.resize_images(image,
                                 size=[resize_height, resize_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  seg_mask = tf.image.resize_images(seg_mask,
                                size=[resize_height, resize_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # Crop to final dimensions.
  #joint_input = tf.concat([image, seg_mask, edge_mask], axis=-1)

  
  if is_training:

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

  tamper_region = tf.clip_by_value(tamper_region, 0.0, 1.0)

  if random_mirror:
  #if False:
    distort=tf.concat([tamper_region, seg_mask,image], axis=-1)
    distort = tf.image.random_flip_left_right(distort,seed=3)
    tamper_region=distort[:,:,:3]
    seg_mask1=distort[:,:,3:4]
    image_m=distort[:,:,4:]
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask1,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask1,tf.float32)) + tamper_region  
    if not is_authentic:
      image_summary("final_tamper", image_distort[:,:,::-1])

      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)

      seg_mask = tf.maximum(seg_mask, seg_mask1)
      return image, seg_mask, image_distort      
    else:
      image_summary("final_tamper", image_distort[:,:,::-1])

      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      return image, seg_mask1, image_distort
    #image = image_m  
  else:
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask,tf.float32))
    image_distort = tf.multiply(image,1-tf.cast(seg_mask,tf.float32)) + tamper_region
    image_summary("final_tamper", image_distort[:,:,::-1])
    #edge_mask = joint_input[:,:,4:]

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

  # Crop to final dimensions.
  #joint_input = tf.concat([image, seg_mask, edge_mask], axis=-1)

  
  if is_training:

    # Randomly mirror the images and labels.

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
    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask1,tf.float32))
    image_distort = tf.multiply(tamper_image,1-tf.cast(seg_mask1,tf.float32)) + tamper_region  
    if not is_authentic:
      image_summary("final_tamper", image_distort[:,:,::-1])
      

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


      image_summary("final_image", image[:,:,::-1])
      #image_summary("final_image", image[:,:,::-1])
      image_summary("final_seg_mask", seg_mask)
      #image_summary("final_edge_mask", edge_mask)

      # Rescale to image-mean instead of [0, 1]
      image_distort = image_distort*255-image_mean
      image = image*255-image_mean

      return image, tamper_image, seg_mask1, image_distort  
  else:

    tamper_region= tf.multiply(tamper_region,tf.cast(seg_mask,tf.float32))
    image_distort = tf.multiply(tamper_image,1-tf.cast(seg_mask,tf.float32)) + tamper_region
    image_summary("final_tamper", image_distort[:,:,::-1])

    image_summary("final_image", image[:,:,::-1])
    image_summary("tamper_image", tamper_image[:,:,::-1])
    image_summary("final_seg_mask", seg_mask)
    #image_summary("final_edge_mask", edge_mask)



    # Rescale to image-mean instead of [0, 1]
    image_distort = image_distort*255-image_mean
    tamper_image = tamper_image*255-image_mean
    image = image*255-image_mean

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


