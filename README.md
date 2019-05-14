# GSRNet
Code for the GSRNet 

# Environment
tensorflow 1.4.0, python3.4, cuda 8.0.44 cudnn 6.0

Other packages please run:
```
pip install -r requirements.txt
```

# Download ImageNet pre-trained model:

Refer to```https://github.com/DrSleep/tensorflow-deeplab-lfov``` for more detail

# Train the model:
Run ```train_default.sh```


# Test the model
1. python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME'

for single image, use --dataset='single_img'

2. save output image:
python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME' --save-dir='./output/$FIXME/' --vis=True --F1=True

3. visualize generated images:
python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME' --save-dir='./output/$FIXME/' --vis=True --vis_gan=True


