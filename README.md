# GSRNet
Code for the GSRNet 

# Environment
tensorflow 0.12.1, python3.4, cuda 8.0.44 cudnn 5.1

Other packages please run:
```
pip install -r requirements.txt
```


# Train the model:
Run ```train_default.sh```


# Test the model
1. python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME'
2. save output image:
python3 dry_run.py --model_weights='./snapshots/$FIXME' --dataset='$FIXME' --save-dir='./output/$FIXME/'

