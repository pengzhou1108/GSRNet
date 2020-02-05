"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os
import sys
import time
import pdb
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import conv,batchnorm,lrelu,deconv,gaussian_noise_layer
from deeplab_resnet import ImageReader, decode_labels, inv_preprocess,inv_preprocess_float, prepare_label
from DeepModel import DeepLabLFOVModel
from scipy import ndimage
from train_ad_tamper_aug_vgg16_fusion_2D import laplacian_im
import glob

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 2



IMG_SIZE=300
add_noise=False
add_jpeg=False
add_scale=False
vis=True
F1=False
AUC=False
vis_gan=False

def remove_isolated_pixels(image,top=5):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    new_image = image.copy()
    if num_stats>0:
      indexes_group = np.argsort(stats[:, cv2.CC_STAT_AREA])
      for component_id, label in enumerate(indexes_group[:-top]):
          new_image[labels == label] = 0

    return new_image

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str, default=None,
                        help="Path to the RGB image file.")
    parser.add_argument("--model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default='./output',
                        help="Where to save predicted mask.")
    parser.add_argument("--single_img", type=str, default='Sp_D_NNN_A_art0084_cha0033_0302.jpg',
                        help="Where to save predicted mask.")
    parser.add_argument("--dataset", type=str, default='coco',
                        help="Where to save predicted mask.")
    parser.add_argument("--vis", type=bool,default=vis)
    parser.add_argument("--vis_gan", type=bool,default=vis_gan)

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


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
      output=convolved
      #output = batchnorm(convolved, is_training)
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
      #output = batchnorm(output, is_training)

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
    
    layers.append(output)

  return layers[-1]


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
    #seg_outputs = image
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
    #edge_loss = tf.losses.mean_squared_error(predictions=tf.multiply(seg_outputs,tf.cast(gt_mask,tf.float32)), labels=tf.multiply(target_im, tf.cast(gt_mask,tf.float32)))
    #edge_loss = tf.losses.mean_squared_error(predictions=tf.multiply(seg_outputs,tf.cast(label_edge,tf.float32)), labels=tf.multiply(gt_image, tf.cast(label_edge,tf.float32)))
    edge_loss = tf.reduce_mean(tf.abs(tf.multiply(seg_outputs,tf.cast(label_edge,tf.float32)) - tf.multiply(gt_image, tf.cast(label_edge,tf.float32))))
    gradient_loss = grad_loss1 + 2*edge_loss
    seg_outputs = tf.image.resize_bilinear(seg_outputs, [h,w])
  return seg_outputs, gen_loss_seg_L1, gradient_loss 

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    if args.dataset=='coco':
      def add_path(path):
          if path not in sys.path:
              sys.path.insert(0, path)
      lib_path = os.path.join('/vulcan/scratch/pengzhou/dataset/PythonAPI')
      add_path(lib_path)
      from pycocotools.coco import COCO
      dataType='train2014'
      annFile='/vulcan/scratch/pengzhou/dataset/PythonAPI/annotations/instances_%s.json'%(dataType)
      coco=COCO(annFile)
      args.img_path='../../dataset/filter_tamper'
      MASK_DIR='../../dataset/casia1_mask'
      data_file='test_filter.txt'

    elif args.dataset=='dvmm':
      args.img_path='/vulcan/scratch/pengzhou/dataset/4cam_splc'
      #data_file='test_dvmm_split.txt'
      data_file='dvmm_seg_whole.txt'

    elif args.dataset=='DSO':

      args.img_path='../../dataset/tifs-database/DSO-1'
      data_file='test_dso.txt'
    elif args.dataset=='COVERAGE':
      args.img_path='../../dataset/COVERAGE'
      data_file='cover_single_seg.txt' 
    else:
      data_file = None

    # Prepare image.
    #img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    
    # Extract mean.   
    img=tf.placeholder(dtype=tf.uint8, shape=(IMG_SIZE, IMG_SIZE, 3))
    seg_mask=tf.placeholder(dtype=tf.uint8, shape=(IMG_SIZE, IMG_SIZE, 1))

    image = tf.cast(img,tf.float32) - IMG_MEAN 
    # Create network.


    if args.vis_gan:
      tar_img=tf.placeholder(dtype=tf.uint8, shape=(IMG_SIZE, IMG_SIZE, 3))
      target_image =tf.cast(tar_img,tf.float32) - IMG_MEAN 
      seg_mask = tf.cast(seg_mask,tf.float32)
      image_cp = tf.multiply(tf.expand_dims(image, dim=0), tf.cast(tf.expand_dims(seg_mask, dim=0),tf.float32)) + tf.multiply(tf.expand_dims(target_image, dim=0), (1-tf.cast(tf.expand_dims(seg_mask, dim=0),tf.float32)))
      with tf.variable_scope("generator") as scope:

          g_output,_, _ =generator(tf.expand_dims(image, dim=0),tf.expand_dims(seg_mask, dim=0),tf.expand_dims(image, dim=0),tf.expand_dims(seg_mask, dim=0),False)



    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE) as scope:
      DeepNet=DeepLabLFOVModel(None) 

      f_net, f_edgenet, f_segnet = DeepNet._create_network(tf.expand_dims(image, dim=0), 1.0, num_classes=args.num_classes, use_fuse=True)
      #f_net = DeepNet._create_network(g_output, 1.0, num_classes=args.num_classes)
      # Predictions.
      f_output = f_net
      f_edge_output = f_edgenet
      f_seg_output = f_segnet

    # Which variables to load.
    #pdb.set_trace()
    restore_var_2 = [v for v in tf.global_variables() if ('discriminator' in v.name)]
    if args.vis_gan:
      restore_var_2 = [v for v in tf.global_variables() if ('generator' in v.name) or ('discriminator' in v.name)]

    # Predictions.
    raw_output = f_output
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    pred_score=tf.nn.softmax(raw_output_up)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    raw_output = f_edge_output
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    edge_pred_score=tf.nn.softmax(raw_output_up)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    edge_pred = tf.expand_dims(raw_output_up, dim=3)

    raw_output = f_seg_output
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    seg_pred_score=tf.nn.softmax(raw_output_up)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    seg_pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var_2)
    load(loader, sess, args.model_weights)



    kernel = np.ones((15,15),np.uint8)
    if data_file:
      f=open(os.path.join(args.img_path,data_file))
    else:
      f=[args.single_img]

    if True:
      for line in f:
          imgname=line.split(' ')[0].strip()
          # Perform inference.
          
          if args.dataset=='coco':
              if os.path.isfile(os.path.join(args.img_path,imgname)):
                  #image_data=cv2.imread(os.path.join(MASK_DIR,'COCO_train2014_{:012d}.jpg'.format(int(imgname.split('_')[1]))))
                  image_data=cv2.imread(os.path.join(args.img_path,imgname))
                  annIds = coco.getAnnIds(imgIds=int(imgname.split('_')[1]), catIds=coco.getCatIds(catNms=[os.path.splitext(imgname)[0].split('_')[-1]]), iscrowd=None)
                  anns = coco.loadAnns(annIds)
                  mask_data=cv2.resize(np.array(coco.annToMask(anns[0])),(image_data.shape[1], image_data.shape[0]))
              else:
                  continue
                  image_data=cv2.imread(os.path.join(MASK_DIR,imgname))
                  mask_data=np.zeros((image_data.shape[0],image_data.shape[1]))

          elif args.dataset=='DSO':

              mask_dir='../../dataset/tifs-database/DSO-1-Fake-Images-Masks'
              image_data=cv2.imread(os.path.join(args.img_path,imgname))
              mask_data=np.logical_not(cv2.imread(os.path.join(mask_dir,imgname))).astype(np.uint8)
              mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2GRAY)
          elif args.dataset=='classification':

              image_data=cv2.imread(imgname)
              mask_data = int(line.split(' ')[1])
          elif args.dataset=='no_mask':

              image_data=cv2.imread(imgname)

              mask_data = np.zeros((IMG_SIZE,IMG_SIZE))
          elif args.dataset=='single_img':

              image_data=cv2.imread(imgname)
              mask_data = np.zeros((IMG_SIZE,IMG_SIZE))
          else:
              image_data=cv2.imread(os.path.join(args.img_path,imgname))
              mask=(cv2.imread(os.path.join(args.img_path,line.strip().split(' ')[1]),cv2.IMREAD_UNCHANGED)>128).astype(np.uint8)
              mask_data = cv2.resize(mask, (IMG_SIZE,IMG_SIZE))
            

          im_h,im_w,_ = image_data.shape
          gt_mask=cv2.resize(mask_data,(IMG_SIZE,IMG_SIZE))
          image_data=cv2.resize(image_data,(IMG_SIZE,IMG_SIZE))

          if args.vis and args.vis_gan:
            if not os.path.exists(args.save_dir):
              os.makedirs(args.save_dir)
            tar_image_data=cv2.imread(tar_img_name)
            tar_image_data=cv2.resize(tar_image_data,(IMG_SIZE,IMG_SIZE))
            #tar_image_data = image_data
            preds,pred_scores, edge_pred_scores,seg_pred_scores, g_output_im,cp_im= sess.run([pred,pred_score,edge_pred_score,seg_pred_score,g_output,image_cp],{img:image_data.astype(np.uint8),tar_img:tar_image_data.astype(np.uint8),seg_mask:gt_mask[:,:,np.newaxis]})

            if not os.path.exists(args.save_dir +'/'+ os.path.splitext(os.path.basename(imgname))[0]+'_gan.png'):
              cv2.imwrite(args.save_dir +'/'+ os.path.basename(imgname),(g_output_im[0]+IMG_MEAN).astype(np.uint8))


          else:
            preds,pred_scores, edge_pred_scores,seg_pred_scores = sess.run([pred,pred_score,edge_pred_score,seg_pred_score],{img:image_data.astype(np.uint8),seg_mask:gt_mask[:,:,np.newaxis]})

          if not os.path.exists(args.save_dir):
              os.makedirs(args.save_dir)


          pred_mask = pred_scores[0,:,:,1]
          edge_pred_mask = edge_pred_scores[0,:,:,1]
          seg_pred_mask = seg_pred_scores[0,:,:,0]
          if args.vis and not args.vis_gan:

            image_data = cv2.resize(image_data,(im_w,im_h))
            #im_seg=(pred_mask>th).astype(np.uint8)
            im_seg= cv2.resize(pred_mask,(im_w,im_h))
            ms_st = time.time()
            #im_seg1=cv2.morphologyEx(im_seg, cv2.MORPH_CLOSE, kernel)
            #im_seg1 = remove_isolated_pixels(im_seg1)
            #edge_seg = (edge_pred_mask>th).astype(np.uint8)
            #seg_seg = (seg_pred_mask>th).astype(np.uint8)
            edge_seg= cv2.resize(edge_pred_mask,(im_w,im_h))
            seg_seg =  cv2.resize(seg_pred_mask,(im_w,im_h))
            gt_mask1 = cv2.resize(gt_mask,(im_w,im_h))
            im_seg4 = im_seg


            fig=plt.figure()
            plt.subplot(151)
            plt.imshow(image_data[:,:,::-1].astype(np.uint8))
            plt.axis('off')
            plt.subplot(152)
            plt.imshow(gt_mask1,cmap='gray')
            
            plt.axis('off')
            plt.subplot(153)
            plt.imshow(edge_seg,cmap='gray')
            plt.axis('off')
            plt.subplot(154)
            plt.imshow(seg_seg,cmap='gray')
            plt.axis('off')
            plt.subplot(155)
            plt.imshow(im_seg4,cmap='gray')
            plt.axis('off')
            fig.savefig(os.path.join(args.save_dir , os.path.splitext(os.path.basename(imgname))[0]+'_cmp.jpg'),bbox_inches='tight')
            plt.close(fig)
            print('The output file has been saved to {}'.format(args.save_dir))


    
if __name__ == '__main__':
    main()

