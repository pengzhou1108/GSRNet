# GSRNet
Code for the GSRNet 

# Environment
tensorflow 1.4.0, python3.4, cuda 8.0.44 cudnn 6.0

Other packages please run:
```
pip install -r requirements.txt
```

# Download ImageNet pre-trained model:

Refer to https://github.com/DrSleep/tensorflow-deeplab-lfov for more detail

Download ```model.ckpt-pretrained```, ```net_skeleton.ckpt``` and put it in the 'ckpt' folder

# Prepare tfrecords for dataset
1. Change the arguments of ```dataset, train_dir, mask_dir, output_directory```, to corresponding directories.

2. Run ```python3 im_pre_casia_pair.py```

3. The tfrecords for COCO and CASIA are provided in
https://drive.google.com/drive/folders/1YY4UM1PBTbBWMyjx350ubp5udGZG66K1?usp=sharing


# Train the model:
1. Change the tfrecords directory in ```train_default.sh```
2. Run ```train_default.sh```


# Test the model
1. python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME'

for single image, use --dataset='single_img'

2. save output image:
python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME' --save-dir='./output/$FIXME/' --vis=True

3. visualize generated images:
python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME' --save-dir='./output/$FIXME/' --vis=True --vis_gan=True

# Citation:
If this code or dataset helps your research, please cite our paper:
```
@inproceedings{zhou2020generate,
  title={Generate, Segment, and Refine: Towards Generic Manipulation Segmentation},
  author={Zhou, Peng and Chen, Bor-Chun and Han, Xintong and Najibi, Mahyar and Shrivastava, Abhinav and Lim, Ser Nam and Davis, Larry S},
  booktitle = {AAAI},
  year={2020}
}
```
