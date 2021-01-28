# LCCNet

![banner]()

![badge]()
![badge]()
[![license](https://img.shields.io/github/license/:user/:repo.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Official PyTorch implementation of the paper “LCCNet: Lidar and Camera Self-Calibration usingCost Volume Network”.

Multi-sensor fusion is for enhancing environment perception and 3D reconstruction in self-driving and robot navigation. Calibration between sensors is the precondition of effective multi-sensor fusion. Laborious manual works and complex environment settings exist in old-fashioned calibration techniques for Light Detection and Ranging (LiDAR) and camera. We propose an online LiDAR-Camera Self-calibration Network (LCCNet), different from the previous CNN-based methods. LCCNet can be trained end-to-end and predict the extrinsic parameters in real-time. In the LCCNet, we exploit the cost volume layer to express the correlation between the features of the RGB image and the depth image projected from point clouds. Besides using the smooth L1-Loss of the predicted extrinsic calibration parameters as a supervised signal, an additional self-supervised signal, point cloud distance loss, is applied during training. Instead of directly regressing the extrinsic parameters, we predict the deviation from initial calibration to the ground truth. The calibration error decreases further with iterative refinement and the temporal filtering approach in the inference stage. The execution time of the calibration process is 24ms for each iteration on a single GPU. LCCNet achieves a mean absolute calibration error of 0.297cm in translation and 0.017° in rotation with miscalibration magnitudes of up to ±1.5m and ±20° on the KITTI-odometry dataset, which is better than the state-of-the-art CNN-based calibration methods.

## Table of Contents

- [Introduction](#Introduction)
  - [Contribution](#Contribution)
  - [Approach overview](#Approach overview)
  - [Main results](#Main results)
- [Requirements](#Requirements)
- [Pre-trained model](#Pre-trained model)
- [Quick test](#Quick test)
- [Dataset prepare](#Dataset prepare)
- [Evaluation](#Evaluation)
- [Train](#Train)
- [Citation](#Citation)
- [Contact](#Contact)
- [License](#license)

## Introduction

### Contribution

### Approach overview

### Main results

## Requirements

* python 3.6 (recommend to use [Anaconda](https://www.anaconda.com/))
* PyTorch==1.0.1.post2
* Torchvision==0.2.2
* Install requirements and dependencies
```angular2html
pip install -r requirements.txt
```

## Pre-trained model

Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1Z6aOqyW1VyzbYW2X7aDOPf3ue_AIlJFB?usp=sharing)

## Quick test

## Dataset prepare

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

## Citation

## Install

This module depends upon a knowledge of [Markdown]().

```
```

### Any optional sections

## Usage

```
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.

### Any optional sections

## API

### Any optional sections

## More optional sections

## Contact

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

### Any optional sections

## License

[MIT © Richard McRichface.](../LICENSE)
