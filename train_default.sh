#!/bin/bash


# train example
python3 train_ad_tamper_aug_vgg16_fusion_2D.py --snapshot-dir='./snapshots/$FIXME' --learning-rate=0.0001 --gan-op='Adam' --not-restore-last --use_refine --use_fuse --not_use_auto --dloss_num=4 --batch-size=4 --tamper-tfrecords='../tensorflow-deeplab-resnet/train_casia2_all_pair_bgr_300.tfrecords' --data-tfrecords='../tensorflow-deeplab-resnet/train_unet_adversarial.tfrecords' --restore-from='../tensorflow-deeplab-resnet/ckpt/model.ckpt-pretrained'

#dry run example
#python3 dry_run.py --model_weights='../tensorflow-deeplab-resnet/snapshots/ad_wgan_noauto_casia_vgg16_fusion_new_coco_rf_bn_mse_b123_4loss_5d_2edge_ginput_300_2D_end2end_d1le-4/model.ckpt-42000' --vis=True --dataset='single_img' --save-dir='./output'


