#!/bin/bash



#GPU_ID=$1
#export CUDA_VISIABLE_DEVICES=${GPU_ID}
python3 train_ad_tamper_aug_vgg16_fusion_2D.py --snapshot-dir='./snapshots/ad_wgan_noauto_casia_vgg16_fusion_new_coco_rf_bn_mse_b123_4loss_5d_2edge_ginput_300_2D_end2end_d1le-4' --learning-rate=0.0001 --gan-op='Adam' --not-restore-last --use_refine --use_fuse --not_use_auto --dloss_num=4 --batch-size=4 --tamper-tfrecords='../tensorflow-deeplab-resnet/train_casia2_all_pair_bgr_300.tfrecords'


