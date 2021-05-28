# LCCNet

Official PyTorch implementation of the paper “LCCNet: Lidar and Camera Self-Calibration Using Cost Volume Network”. A video of the demonstration of the method can be found on
 https://www.youtube.com/watch?v=UAAGjYT708A

## Table of Contents

- [Requirements](#Requirements)
- [Pre-trained model](#Pre-trained_model)
- [Evaluation](#Evaluation)
- [Train](#Train)
- [Citation](#Citation)



## Requirements

* python 3.6 (recommend to use [Anaconda](https://www.anaconda.com/))
* PyTorch==1.0.1.post2
* Torchvision==0.2.2
* Install requirements and dependencies
```commandline
pip install -r requirements.txt
```

## Pre-trained model

Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1Z6aOqyW1VyzbYW2X7aDOPf3ue_AIlJFB?usp=sharing)

## Evaluation

1. Download [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. Change the path to the dataset in `evaluate_calib.py`.
```python
data_folder = '/path/to/the/KITTI/odometry_color/'
```
3. Create a folder named `pretrained` to store the pre-trained models in the root path.
4. Download pre-trained models and modify the weights path in `evaluate_calib.py`.
```python
weights = [
   './pretrained/kitti_iter1.tar',
   './pretrained/kitti_iter2.tar',
   './pretrained/kitti_iter3.tar',
   './pretrained/kitti_iter4.tar',
   './pretrained/kitti_iter5.tar',
]
```
5. Run evaluation.
```commandline
python evaluate_calib.py
```

## Train
```commandline
python train_with_sacred.py
```

## Citation
 
Thank you for citing our paper if you use any of this code or datasets.
```
@article{lv2020lidar,
  title={Lidar and Camera Self-Calibration using CostVolume Network},
  author={Lv, Xudong and Wang, Boya and Ye, Dong and Wang, Shuo},
  journal={arXiv preprint arXiv:2012.13901},
  year={2020}
}
```

---
