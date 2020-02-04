"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
from utils import prefetch_input_data, process_image, process_image_pair, parse_tf_example,parse_tf_example_pair, conv,batchnorm_old,lrelu,deconv,focal_loss
import cv2
import tensorflow as tf
import numpy as np
import pdb
from deeplab_resnet import ImageReader, decode_labels, inv_preprocess,inv_preprocess_float, prepare_label
from DeepModel import DeepLabLFOVModel

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 4
DATA_DIRECTORY = '../../dataset/CASIA2/'
MASK_DIRECTORY = '../../dataset/casia2_mask/'
DATA_LIST_PATH = '../../dataset/CASIA2/train_all_single_seg.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '300,300'
#INPUT_SIZE = '256,256'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 60000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './ckpt/model.ckpt-pretrained'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 3000
SNAPSHOT_DIR = './snapshots/adversarial_lr_0001_wgan_addnoise_w2tg05_casia'
WEIGHT_DECAY = 0.00005
DATA_TFRECORDS = './train_unet_adversarial.tfrecords'
TAMPER_SAMPLE = '../Unet/data/train_unet_coco_256.tfrecords'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument('--dloss_num', type=int, default=3)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-tfrecords", type=str, default=DATA_TFRECORDS,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--tamper-tfrecords", type=str, default=TAMPER_SAMPLE,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--mask-dir", type=str, default=MASK_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", type=bool,default=True,
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--g_checkpoint", type=str, default=None,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gan-op", type=str, default='wgan',
                        help="gan loss model.")
    parser.add_argument("--use_floss", type=bool, default=False,
                        help="use focal loss")
    parser.add_argument("--use_refine", action="store_true",
                        help="use refine loss")
    parser.add_argument("--use_fuse", action="store_true",
                        help="use fusion")
    parser.add_argument("--not_use_auto", action="store_false",
                        help="use refine loss")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
   print('The checkpoint has been created.')

def build_input(args,is_authentic=True, queue_name='authentic_queue', input_queue_name='authentic_input_queue'):
  # Load input data
  if is_authentic:
    input_queue = prefetch_input_data(
      tf.TFRecordReader(),
      args.data_tfrecords,
      is_training=args.is_training,
      batch_size=args.batch_size,
      values_per_shard=2000,
      input_queue_capacity_factor=2,
      num_reader_threads=1,
      shard_queue_name=queue_name,
      value_queue_name=input_queue_name)
  else:
    input_queue = prefetch_input_data(
      tf.TFRecordReader(),
      args.tamper_tfrecords,
      is_training=args.is_training,
      batch_size=args.batch_size,
      values_per_shard=2000,
      input_queue_capacity_factor=2,
      num_reader_threads=1,
      shard_queue_name=queue_name,
      value_queue_name=input_queue_name)

  # Image processing and random distortion. Split across multiple threads
  # with each thread applying a slightly different distortion.
  # assert self.config.num_preprocess_threads % 2 == 0
  images_and_maps = []
  h, w = map(int, args.input_size.split(','))
  for thread_id in range(1):
    serialized_example = input_queue.dequeue()
    if not is_authentic:
      (image, tamper_im, seg_mask, image_id, label) = parse_tf_example_pair(serialized_example)
      (image, tamper_im, seg_mask,image_distort) = process_image_pair(image,
                                      tamper_im,
                                      seg_mask,
                                      args.is_training,
                                      (args.random_mirror and is_authentic),
                                      args.random_scale,
                                      IMG_MEAN,
                                      height=h,
                                      width=w,
                                      resize_height=h,
                                      resize_width=w)
    else:
      (image, seg_mask, image_id, label) = parse_tf_example(serialized_example)
      (image, seg_mask,image_distort) = process_image(image,
                                    seg_mask,
                                    args.is_training,
                                    (args.random_mirror and is_authentic),
                                    args.random_scale,
                                    IMG_MEAN,
                                    height=256,
                                    width=256,
                                    resize_height=h,
                                    resize_width=w) 
      tamper_im = image        
    images_and_maps.append([image, tamper_im, seg_mask, image_id,image_distort, label])


  # Batch inputs.
  queue_capacity = (5 * 1 *
                    args.batch_size)

  return tf.train.batch_join(images_and_maps,
                             batch_size=args.batch_size,
                             capacity=queue_capacity,
                             name=queue_name)

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                               tf.random_normal_initializer(stddev=stddev))
      bias = tf.get_variable("bias", [output_size],
          initializer=tf.constant_initializer(bias_start))
      if with_w:
          return tf.matmul(input_, matrix) + bias, matrix, bias
      else:
          return tf.matmul(input_, matrix) + bias
def create_generator(image, generator_outputs_channels, is_training=True):
  """ Generator from image to a segment map"""
  # Build inputs
  layers = []

  # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
  with tf.variable_scope("encoder_1"):
    output = conv(image, 64, stride=2)
    layers.append(output)

    layer_specs = [
      # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
      64 * 2,
      # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
      64 * 4,
      # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
      64 * 8,
      # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
      64 * 8,
      # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
      64 * 8,
      # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
      64 * 8,
      # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
      64 * 8,
  ]

  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height/2, in_width/2, out_channels]
      convolved = conv(rectified, out_channels, stride=2)
      output = convolved
      #output = batchnorm_old(convolved, is_training)
      layers.append(output)

  layer_specs = [
    # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    (64 * 8, 0.0),
    # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    (64 * 8, 0.0),
    # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    (64 * 8, 0.0),
    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    (64 * 8, 0.0),
    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    (64 * 4, 0.0),
    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    (64 * 2, 0.0),
    # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    (64, 0.0),
  ]


  num_encoder_layers = len(layers)
  for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
    skip_layer = num_encoder_layers - decoder_layer - 1
    with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
      if decoder_layer == 0:
        # first decoder layer doesn't have skip connections
        # since it is directly connected to the skip_layer
        input = layers[-1]
      else:
        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

      rectified = tf.nn.relu(input)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height*2, in_width*2, out_channels]
      output = deconv(rectified, out_channels)
      #output = batchnorm_old(output, is_training)

      if dropout > 0.0 and is_training:
        output = tf.nn.dropout(output, keep_prob=1 - dropout)

      layers.append(output)

  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256,
  # generator_outputs_channels]
  with tf.variable_scope("decoder_1"):
    input = tf.concat([layers[-1], layers[0]], axis=3)
    rectified = tf.nn.relu(input)
    output = deconv(rectified, generator_outputs_channels)
    output = tf.tanh(output)
    # output = tf.sigmoid(output)
    layers.append(output)

  return layers[-1]


def create_discriminator(image,args, is_training=True):
  """ Generator from image to a segment map"""
  # Build inputs
  layers = []
  image = tf.image.resize_bilinear(image, [256, 256])
  # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
  with tf.variable_scope("encoder_0"):
    output = conv(image, 64, stride=2)
    layers.append(output)

    layer_specs = [
      # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
      (64 * 2, 0.0),
      # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
      (64 * 4,0.0),
      # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 31, 31, ngf * 8]
      (64 * 8,0.0)
  ]

  for decoder_layer, (out_channels,dropout) in enumerate(layer_specs):
    with tf.variable_scope("encoder_%d" % (decoder_layer+1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels]
      # => [batch, in_height/2, in_width/2, out_channels]
      if decoder_layer==len(layer_specs)-1:
        convolved = conv(rectified, out_channels, stride=1)
      else:
        convolved = conv(rectified, out_channels, stride=2)
      output=convolved
      #output = batchnorm_old(convolved, is_training)
      if dropout > 0.0 and is_training:
        output = tf.nn.dropout(output, keep_prob=1 - dropout)
      layers.append(output)
  with tf.variable_scope("encoder_%d" % (len(layer_specs)+1)):
    rectified = lrelu(layers[-1], 0.2)
    # [batch, in_height, in_width, in_channels]
    # => [batch, in_height/2, in_width/2, out_channels]
    convolved = conv(rectified, 1, stride=1)
    output=tf.nn.sigmoid(convolved)
    layers.append(output)

  return layers[-1]


def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = a.reshape(list(a.shape) + [1,1])
  return tf.tile(tf.constant(a, dtype=1),[1,1,3,1])
def laplacian_im(im):
  laplacian = make_kernel(np.asarray([[0,1,0],[1,-4,1],[0,1,0]])/4)
  y = tf.nn.depthwise_conv2d(im, laplacian, [1, 1, 1, 1], padding='SAME')
  return y
def generator(gt_image, gt_mask, target_im, target_mask, is_training=False):
  """Build the model given tampered image predict tampered edge segmentation.
  """
  h,w = gt_image.get_shape()[1:3]
  with tf.variable_scope("generator") as scope:
    joint_input = tf.concat([gt_image, tf.cast(gt_mask,tf.float32),target_im], axis=-1)  
    joint_input = tf.image.resize_bilinear(joint_input, [256, 256])
    gt_image = joint_input[:,:,:,:3]
    gt_mask = joint_input[:,:,:,3:4]
    target_im = joint_input[:,:,:,4:7]
    image = tf.multiply(gt_image, tf.cast(gt_mask,tf.float32)) + tf.multiply(target_im, (1-tf.cast(gt_mask,tf.float32)))

    out_channels = int(gt_image.get_shape()[-1])

    outputs = create_generator(tf.concat([image, tf.cast(gt_mask,tf.float32)], axis=-1), out_channels, is_training)

    seg_outputs = outputs[:,:,:,0:3]

    seg_outputs = (seg_outputs/2+0.5)*255-IMG_MEAN

  with tf.name_scope("generator_loss"):
    # output image
    gen_loss_seg_L1 = tf.reduce_mean(tf.abs(tf.multiply(seg_outputs,(1-tf.cast(gt_mask,tf.float32))) - tf.multiply(target_im,(1-tf.cast(gt_mask,tf.float32)))))
    grad_loss1 = tf.reduce_mean(tf.abs(tf.multiply(laplacian_im(seg_outputs),tf.cast(gt_mask,tf.float32)) - tf.multiply(laplacian_im(gt_image), tf.cast(gt_mask,tf.float32))))
    # edge
    kernel_tensor = tf.constant(np.ones((5,5,1),np.float32),name='kernel_tensor')
    label_erosion=tf.nn.erosion2d(tf.cast(gt_mask,tf.float32),kernel_tensor,
                                 strides=[1,1,1,1],
                                 rates=[1,1,1,1],
                                 padding='SAME')
    label_dilation=tf.nn.dilation2d(tf.cast(gt_mask,tf.float32),kernel_tensor,
                                 strides=[1,1,1,1],
                                 rates=[1,1,1,1],
                                 padding='SAME')
    label_edge = label_dilation - label_erosion -2
    edge_loss = tf.reduce_mean(tf.abs(tf.multiply(seg_outputs,tf.cast(label_edge,tf.float32)) - tf.multiply(gt_image, tf.cast(label_edge,tf.float32))))
    gradient_loss = grad_loss1 + 2*edge_loss
    seg_outputs = tf.image.resize_bilinear(seg_outputs, [h,w])
  return seg_outputs, gen_loss_seg_L1, gradient_loss 


def discriminator(image, args, is_training=True):
  """Build the model given tampered image predict tampered edge segmentation.
  """

    #out_channels = int(seg_mask.get_shape()[-1] + edge_mask.get_shape()[-1])
  d_outputs = create_discriminator(image, args, is_training)

  return d_outputs

def main():
    """Create the model and start the training."""
    args = get_arguments()
    if args.is_training:
      print('is training')

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    tf.set_random_seed(args.random_seed)
    print('dloss num is {:d}'.format(args.dloss_num))
    # Create queue coordinator.
    coord = tf.train.Coordinator()   
    # Load reader.
    with tf.name_scope("create_inputs"):
        #gt_image_batch, tamper_batch, label_batch, image_ids, gt_distort_batch,_ = build_input(args,is_authentic=False)
        gt_image_batch, tamper_batch, label_batch, image_ids, gt_distort_batch,_ = build_input(args,is_authentic=True)
        #tamper_batch,tamper_label_batch, image_ids, image_batch, _ =build_input(args, is_authentic=False)
        target_gt_batch, target_batch,target_label_batch, target_image_ids, target_distort_batch, _ =build_input(args, is_authentic=False, queue_name='target_queue',input_queue_name='target_input_queue')

        label_batch=tf.zeros(label_batch.get_shape(),tf.int32)

        target_label_batch=tf.cast(target_label_batch, tf.int32)

    # Create network.
    with tf.variable_scope("generator") as scope:
      if args.use_refine:

        g_output, ge_loss, grad_loss=generator(target_batch, target_label_batch, gt_image_batch,label_batch,args.is_training)
      else:
        g_output, ge_loss, grad_loss=generator(target_batch, target_label_batch, gt_image_batch,label_batch,args.is_training)

    if args.gan_op!='wgan':
      with tf.variable_scope("discriminator1") as scope:
        d1_fake=discriminator(tf.concat([g_output, tf.cast(target_label_batch,tf.float32)], axis=-1), args, args.is_training)
      with tf.variable_scope("discriminator1",reuse=True) as scope:
        d1_real=discriminator(tf.concat([gt_image_batch, tf.cast(target_label_batch,tf.float32)], axis=-1), args, args.is_training)

    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE) as scope:
      DeepNet=DeepLabLFOVModel(None) 
      if args.use_fuse:
        f_net, f_edgenet, _ = DeepNet._create_network(g_output, 0.5, num_classes=args.num_classes, use_fuse=True)
        # Predictions.
        f_output = f_net
        f_edge = f_edgenet
      else:
        f_net = DeepNet._create_network(g_output, 0.5, num_classes=args.num_classes)
        # Predictions.
        f_output = f_net
       
      if args.dloss_num>3:
        if args.use_fuse:
          t_input = tf.multiply(target_batch, tf.cast(target_label_batch,tf.float32)) + tf.multiply(gt_image_batch, (1-tf.cast(target_label_batch,tf.float32)))
          t_net, t_edgenet,_ = DeepNet._create_network(t_input, 0.5, num_classes=args.num_classes, use_fuse=True)
          t_output = t_net
          t_edge_output = t_edgenet
        else:
          t_input = tf.multiply(target_batch, tf.cast(target_label_batch,tf.float32)) + tf.multiply(gt_image_batch, (1-tf.cast(target_label_batch,tf.float32)))
          t_net = DeepNet._create_network(t_input, 0.5, num_classes=args.num_classes)
          t_output = t_net       

      #t_output = t_net
      if args.not_use_auto:
        if args.use_fuse:
          tar_net,tar_edgenet, _ = DeepNet._create_network(target_auto, 0.5, num_classes=args.num_classes, use_fuse=True)
          tar_output = tar_net
          tar_edge_output = tar_edgenet
        else:
          tar_net = DeepNet._create_network(target_auto, 0.5, num_classes=args.num_classes)
          tar_output = tar_net
      else:
        if args.use_fuse:
          tar_net,tar_edgenet,_ = DeepNet._create_network(target_batch, 0.5, num_classes=args.num_classes, use_fuse=True)
          tar_output = tar_net
          tar_edge_output = tar_edgenet
        else:
          tar_net = DeepNet._create_network(target_batch, 0.5, num_classes=args.num_classes)
          tar_output = tar_net        

      if args.use_refine:
        # output refine
        if args.use_fuse:
          f_output_up = tf.image.resize_bilinear(tar_edge_output, tf.shape(target_batch)[1:3,])
        else:
          f_output_up = tf.image.resize_bilinear(tar_output, tf.shape(target_batch)[1:3,])
        f_output_up = tf.expand_dims(tf.argmax(f_output_up, dimension=3), -1)
        f_refine = tf.multiply(target_batch,1-tf.cast(f_output_up, tf.float32)) + tf.multiply(target_gt_batch, tf.cast(f_output_up, tf.float32))
        if args.use_fuse:
          f_net_refine, f_net_refine_edge,_ = DeepNet._create_network(f_refine, 0.5, num_classes=args.num_classes, use_fuse=True)
          f_refine_output = f_net_refine
          f_refine_edge_output = f_net_refine_edge
        else:
          f_net_refine = DeepNet._create_network(f_refine, 0.5, num_classes=args.num_classes)
          f_refine_output = f_net_refine 
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.


    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    #restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    restore_var={}
    
    for v in tf.global_variables():
      if v.name.split('/')[0]=='discriminator' and (('fc' not in v.name and 'batchnorm' not in v.name) or not args.not_restore_last):
        restore_var[v.name.split(':')[0].replace('discriminator/','',1)]=v

    D_all_trainable = [v for v in tf.trainable_variables() if (v.name.split('/')[0]=='discriminator') or (v.name.split('/')[0]=='generator')]

    D1_trainable = [v for v in tf.trainable_variables() if v.name.split('/')[0]=='discriminator1']


    # edge
    kernel_tensor = tf.constant(np.ones((5,5,1),np.float32),name='kernel_tensor')
    label_erosion=tf.nn.erosion2d(tf.cast(label_batch,tf.float32),kernel_tensor,
                                 strides=[1,1,1,1],
                                 rates=[1,1,1,1],
                                 padding='SAME')
    label_dilation=tf.nn.dilation2d(tf.cast(label_batch,tf.float32),kernel_tensor,
                                 strides=[1,1,1,1],
                                 rates=[1,1,1,1],
                                 padding='SAME')
    tar_label_erosion=tf.nn.erosion2d(tf.cast(target_label_batch,tf.float32),kernel_tensor,
                                 strides=[1,1,1,1],
                                 rates=[1,1,1,1],
                                 padding='SAME')
    tar_label_dilation=tf.nn.dilation2d(tf.cast(target_label_batch,tf.float32),kernel_tensor,
                                 strides=[1,1,1,1],
                                 rates=[1,1,1,1],
                                 padding='SAME')
    label_edge = label_dilation - label_erosion -2
    target_edge = tar_label_dilation - tar_label_erosion -2
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
 


    f_label_batch_seg = tf.cast(tf.maximum(label_batch, target_label_batch), tf.int32)
    raw_prediction = tf.reshape(f_output, [-1, args.num_classes])
    label_proc = prepare_label(f_label_batch_seg, tf.stack(f_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    f_raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(f_raw_gt, args.num_classes - 1)), 1)
    f_gt = tf.cast(tf.gather(f_raw_gt, indices), tf.int32)
    f_prediction = tf.gather(raw_prediction, indices)
    if args.use_fuse:
      f_label_batch = tf.cast(tf.maximum(label_edge, target_edge), tf.int32)
      raw_prediction = tf.reshape(f_edge, [-1, args.num_classes])
      label_proc = prepare_label(f_label_batch, tf.stack(f_edge.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
      f_raw_gt = tf.reshape(label_proc, [-1,])
      indices = tf.squeeze(tf.where(tf.less_equal(f_raw_gt, args.num_classes - 1)), 1)
      f_edge_gt = tf.cast(tf.gather(f_raw_gt, indices), tf.int32)
      f_edge_prediction = tf.gather(raw_prediction, indices)  

    if args.use_refine:  

      refine_label=tf.cast(target_label_batch, tf.float32) - tf.cast(target_label_batch, tf.float32) * tf.cast(f_output_up, tf.float32)
      refine_label_erosion=tf.nn.erosion2d(tf.cast(refine_label,tf.float32),kernel_tensor,
                                   strides=[1,1,1,1],
                                   rates=[1,1,1,1],
                                   padding='SAME')
      refine_label_dilation=tf.nn.dilation2d(tf.cast(refine_label,tf.float32),kernel_tensor,
                                   strides=[1,1,1,1],
                                   rates=[1,1,1,1],
                                   padding='SAME')
      refine_label_edge = refine_label_dilation - refine_label_erosion -2  
      raw_prediction = tf.reshape(f_refine_output, [-1, args.num_classes])    
      label_proc = prepare_label(refine_label, tf.stack(f_refine_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
      f_raw_gt = tf.reshape(label_proc, [-1,])
      indices = tf.squeeze(tf.where(tf.less_equal(f_raw_gt, args.num_classes - 1)), 1)
      f_refine_gt = tf.cast(tf.gather(f_raw_gt, indices), tf.int32)
      f_refine_prediction = tf.gather(raw_prediction, indices) 

      if args.use_fuse:
        # edge refine
        raw_prediction = tf.reshape(f_refine_edge_output, [-1, args.num_classes])
        label_proc = prepare_label(refine_label_edge, tf.stack(f_refine_edge_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
        f_raw_gt = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(f_raw_gt, args.num_classes - 1)), 1)
        f_refine_edge_gt = tf.cast(tf.gather(f_raw_gt, indices), tf.int32)
        f_refine_edge_prediction = tf.gather(raw_prediction, indices) 



    raw_prediction = tf.reshape(tar_output, [-1, args.num_classes])
    label_proc = prepare_label(target_label_batch, tf.stack(target_batch.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    tar_gt_raw = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(tar_gt_raw, args.num_classes - 1)), 1)
    tar_gt = tf.cast(tf.gather(tar_gt_raw, indices), tf.int32)
    tar_prediction = tf.gather(raw_prediction, indices) 

    if args.use_fuse:
      # edge
      label_proc = prepare_label(target_edge, tf.stack(target_batch.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
      raw_prediction = tf.reshape(tar_edge_output, [-1, args.num_classes])
      tar_edge_gt_raw = tf.reshape(label_proc, [-1,])
      indices = tf.squeeze(tf.where(tf.less_equal(tar_edge_gt_raw, args.num_classes - 1)), 1)
      tar_edge_gt = tf.cast(tf.gather(tar_edge_gt_raw, indices), tf.int32)
      tar_edge_prediction = tf.gather(raw_prediction, indices)  




    if args.dloss_num>3:
    # tamper prediction
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
      raw_prediction = tf.reshape(t_output, [-1, args.num_classes])
      label_proc = prepare_label(target_label_batch, tf.stack(t_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
      raw_gt = tf.reshape(label_proc, [-1,])
      indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
      t_gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
      t_prediction = tf.gather(raw_prediction, indices) 
      if args.use_fuse:
        # edge
        raw_prediction = tf.reshape(t_edge_output, [-1, args.num_classes])
        label_proc = prepare_label(target_edge, tf.stack(t_edge_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
        
        t_edge_gt_raw = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(t_edge_gt_raw, args.num_classes - 1)), 1)
        t_edge_gt = tf.cast(tf.gather(t_edge_gt_raw, indices), tf.int32)
        t_edge_prediction = tf.gather(raw_prediction, indices)                                                      
                      
    if args.use_refine:

      Df_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_prediction, labels=f_gt))
      Df_refine_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_refine_prediction, labels=f_refine_gt))
      Dgt_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tar_prediction, labels=tar_gt))

      D_l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('w:0' in v.name or 'filter' in v.name) and v.name.split('/')[0]=='discriminator']
      Dgt_edge_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=tar_edge_prediction, labels=tar_edge_gt, weights=tf.gather([0.3, 0.7], tar_edge_gt)))
      Df_refine_edge_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=f_refine_edge_prediction, labels=f_refine_edge_gt, weights=tf.gather([0.3, 0.7], f_refine_edge_gt)))
      Df_edge_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=f_edge_prediction, labels=f_edge_gt, weights=tf.gather([0.3, 0.7], f_edge_gt)))

      D_loss =  Df_loss  + Df_refine_loss  + Dgt_loss +  Df_edge_loss + Df_refine_edge_loss + Dgt_edge_loss + tf.add_n(D_l2_losses)
    elif args.use_fuse:
      Df_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_prediction, labels=f_gt))
      Df_edge_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_edge_prediction, labels=f_edge_gt))
      Dgt_edge_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tar_edge_prediction, labels=tar_edge_gt))
      Dgt_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tar_prediction, labels=tar_gt))
      D_l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('w:0' in v.name or 'filter' in v.name) and v.name.split('/')[0]=='discriminator']
      D_loss =  Df_loss  + Dgt_loss +  tf.add_n(D_l2_losses) +  Df_edge_loss + Dgt_edge_loss
    else:
      Df_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=f_prediction, labels=f_gt))
      Dgt_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tar_prediction, labels=tar_gt))
      D_l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('w:0' in v.name or 'filter' in v.name) and v.name.split('/')[0]=='discriminator']
      D_loss =  Df_loss  + Dgt_loss +  tf.add_n(D_l2_losses)  
          
    if args.dloss_num>3:
      Dt_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=t_prediction, labels=t_gt))
      if args.use_fuse:
        Dt_edge_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=t_edge_prediction, labels=t_edge_gt, weights=tf.gather([0.3, 0.7], t_edge_gt)))
        D_loss = D_loss + Dt_loss + Dt_edge_loss
      else:
        D_loss = D_loss + Dt_loss
    if args.gan_op=='wgan':
      G_loss =  ge_loss + grad_loss#- 100*tf.reduce_mean(f_h)
    else:
      
      D1_loss = tf.reduce_mean(-(tf.log(d1_real + 1e-9) + tf.log(1 - d1_fake + 1e-9)))
      gf_loss = tf.reduce_mean(-tf.log(d1_fake + 1e-9))
      G_loss =  ge_loss + grad_loss + 5*gf_loss
      
      D_loss += G_loss
    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(f_output, tf.shape(target_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    if args.use_fuse:
      raw_output_up = tf.image.resize_bilinear(tar_edge_output, tf.shape(target_batch)[1:3,])
      raw_output_up = tf.argmax(raw_output_up, dimension=3)
      tar_edge_pred = tf.expand_dims(raw_output_up, dim=3)
      tar_edge_summary = tf.py_func(decode_labels, [tar_edge_pred, args.save_num_images, args.num_classes], tf.uint8)

    if args.use_refine:

      f_refine_images_summary = tf.py_func(inv_preprocess, [f_refine, args.save_num_images,IMG_MEAN], tf.uint8)
      refine_labels_summary = tf.py_func(decode_labels, [tf.cast(refine_label, tf.int32), args.save_num_images, args.num_classes], tf.uint8)
      raw_output_up = tf.image.resize_bilinear(f_refine_output, tf.shape(target_batch)[1:3,])
      raw_output_up = tf.argmax(raw_output_up, dimension=3)
      f_refine_pred = tf.expand_dims(raw_output_up, dim=3)    

      f_refine_preds_summary = tf.py_func(decode_labels, [f_refine_pred, args.save_num_images, args.num_classes], tf.uint8)

      Df_refine_loss_summary=tf.summary.scalar('D_f_refine_loss',Df_refine_loss)
      
    raw_output_up = tf.image.resize_bilinear(tar_output, tf.shape(target_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    tar_pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [target_batch, args.save_num_images,IMG_MEAN], tf.uint8)
    target_summary = tf.py_func(inv_preprocess, [gt_image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    g_output_summary = tf.py_func(inv_preprocess, [g_output, args.save_num_images,IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [f_label_batch_seg, args.save_num_images, args.num_classes], tf.uint8)
    #gt_labels_summary = tf.py_func(decode_labels, [gt_label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)

    #refine pred

    tar_summary = tf.py_func(decode_labels, [tar_pred, args.save_num_images, args.num_classes], tf.uint8)
    #tamper pred
    if args.dloss_num>3:
      t_summary = tf.py_func(inv_preprocess, [t_input, args.save_num_images,IMG_MEAN], tf.uint8)
      #t_labels_summary = tf.py_func(decode_labels, [tamper_label_batch, args.save_num_images, args.num_classes], tf.uint8)
      raw_output_up = tf.image.resize_bilinear(t_output, tf.shape(target_batch)[1:3,])
      raw_output_up = tf.argmax(raw_output_up, dimension=3)
      t_pred = tf.expand_dims(raw_output_up, dim=3)
      t_preds_summary = tf.py_func(decode_labels, [t_pred, args.save_num_images, args.num_classes], tf.uint8)
    
    if args.use_refine:
      if args.dloss_num>3:
        total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary,target_summary,t_summary, g_output_summary,labels_summary, preds_summary, f_refine_images_summary,refine_labels_summary, f_refine_preds_summary, tar_summary, tar_edge_summary, t_preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
      else:
        total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary,target_summary, g_output_summary,labels_summary, preds_summary, f_refine_images_summary,refine_labels_summary, f_refine_preds_summary, tar_summary, tar_edge_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    else:
      if args.use_fuse:
        total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary,target_summary, g_output_summary,labels_summary, preds_summary, tar_summary, tar_edge_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.    
      else:
        total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary,target_summary, g_output_summary,labels_summary, preds_summary, tar_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.             

    D_loss_summary=tf.summary.scalar('D_loss',D_loss)
    if args.gan_op!='wgan':
      D1_loss_summary=tf.summary.scalar('D1_loss',D1_loss)
    if args.dloss_num>=4:
      Dgt_loss_summary=tf.summary.scalar('Dt_loss',Dt_loss)
    Df_loss_summary=tf.summary.scalar('D_f_loss',Df_loss)
    Ge_loss_summary=tf.summary.scalar('G_l2_loss',ge_loss)
    Grad_loss_summary=tf.summary.scalar('Grad_loss',grad_loss)
    G_loss_summary=tf.summary.scalar('G_loss',G_loss)
    summary_all = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # Define loss and optimisation parameters. Discriminator
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    

    # Define loss and optimisation parameters. Generator
    #learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    if args.gan_op=='SGD':
      G_opt_g = tf.train.MomentumOptimizer(learning_rate*10, args.momentum)

      G_grads = tf.gradients(G_loss, G_conv_trainable + conv_trainable + fc_w_trainable + fc_b_trainable)
      G_grads_g = G_grads[:len(G_conv_trainable)]



      G_train_op_g = G_opt_g.apply_gradients(zip(G_grads_g, G_conv_trainable))

      G_train_op=G_train_op_g




      D1_opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)



      D1_grads = tf.gradients(D1_loss,G_conv_trainable + D1_trainable)

      D1_grads_conv = D1_grads[len(G_conv_trainable):]


      D1_train_op_conv = D1_opt_conv.apply_gradients(zip(D1_grads_conv, D1_trainable))

      D1_train_op = D1_train_op_conv
    elif args.gan_op=='Adam':
      D_train_op = tf.train.AdamOptimizer(args.learning_rate) \
                            .minimize(D_loss, var_list=D_all_trainable) 

      D1_train_op = tf.train.AdamOptimizer(5e-5, beta1=0.5) \
                          .minimize(D1_loss, var_list=D1_trainable)

    elif args.gan_op=='wgan':
      D_train_op = tf.train.AdamOptimizer(args.learning_rate) \
                            .minimize(D_loss, var_list=D_all_trainable) 
      G_train_op = tf.train.AdamOptimizer(5e-5, beta1=0.5) \
                            .minimize(G_loss, var_list=G_conv_trainable)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=30)
    step=0
    # Load variables if the checkpoint is provided.
    if args.checkpoint is not None:
      checkpoint = tf.train.latest_checkpoint(args.checkpoint)
      print('restore from {:s}'.format(args.checkpoint))
      saver.restore(sess,checkpoint)
      step += int(checkpoint.split('-')[1])
    elif tf.train.latest_checkpoint(args.snapshot_dir) is not None:
      checkpoint = tf.train.latest_checkpoint(args.snapshot_dir)
      print('restore from the last training {:s}'.format(args.snapshot_dir))
      sys.stdout.flush()
      saver.restore(sess, checkpoint)
      step += int(checkpoint.split('-')[1])
    elif args.restore_from is not None:
      #loader = tf.train.Saver(var_list=restore_var)
      loader = tf.train.Saver(restore_var)
      load(loader, sess, args.restore_from)
    if args.g_checkpoint is not None:
      g_checkpoint = tf.train.latest_checkpoint(args.g_checkpoint)
      g_restore_var = [v for v in tf.trainable_variables() if (v.name.split('/')[0]=='generator' or v.name.split('/')[0]=='discriminator1') ]
      g_loader = tf.train.Saver(g_restore_var)

      load(g_loader, sess, g_checkpoint)


    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    while step < args.num_steps:
        start_time = time.time()
        feed_dict = { step_ph : step }
        step +=1
        if step % 100 == 0:
          D1_loss_value, preds, summary, _ = sess.run([D1_loss, pred, summary_all, D1_train_op], feed_dict=feed_dict)
          #D_loss_value, preds, summary, _ = sess.run([D_loss, pred, summary_all, D_train_op], feed_dict=feed_dict)
          D_loss_value, preds, summary, _ = sess.run([D_loss, pred, summary_all, D_train_op], feed_dict=feed_dict)
          #G_loss_value, preds, summary, _ = sess.run([G_loss, pred, summary_all, G_train_op], feed_dict=feed_dict)
          summary_writer.add_summary(summary, step)

          if step % args.save_pred_every == 0:
            save(saver, sess, args.snapshot_dir, step)
        else:            
            D1_loss_value, _ = sess.run([D1_loss, D1_train_op], feed_dict=feed_dict)

            D_loss_value, _ = sess.run([D_loss, D_train_op], feed_dict=feed_dict)
            #summary_writer.add_summary(G_loss_value, step)
        duration = time.time() - start_time
        if args.gan_op=='wgan':

          print('step {:d} \t Dloss = {:.3f}, ({:.3f} sec/step)'.format(step, D_loss_value, duration))
          sys.stdout.flush()
        else:
          print('step {:d} \t Dloss = {:.3f}, D1loss = {:.3f}, ({:.3f} sec/step)'.format(step, D_loss_value, D1_loss_value, duration))
          sys.stdout.flush()          
        
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    if not os.path.isdir(SNAPSHOT_DIR):
      os.mkdir(SNAPSHOT_DIR)
    main()
