# Visual-Perception-based-Navigation
## Introduction
**Authors:** T. Ran, L. Yuan, J. Zhang

<img src="https://github.com/rantengsky/Visual-Perception-based-Navigation/blob/master/pictures/fig1.png" width="375">

This project is a real-time indoor navigation system for the mobile robot. It takes the information of an monocular camera as input. The images captured by the camera will be classified by a CNN model. After the classification, the system converts the outputs to specific velocity command through the motion control algorithm (the regular control and the adaptive weighted control). And the robot can navigate by itself under an unknown and dynamic environment.

**Video:** [nav-video](http://i.youku.com/i/UNjgyMjQ5NzM0NA==/videos?spm=a2hzp.8244740.0.0)

## Prerequisites

### 1. Window 10

### 2. Python 3.X

### 3. TensorFlow-GPU 1.10

### 4. OpenCv 3.4.X

### 5. DataBase: MySQL

To store the encoder data (pose, velocity)

## Contact us

For any issues, please feel free to contact Teng Ran: rantengsky@163.com

## Citation
BibTex
@article{RAN2021389,
title = {Scene perception based visual navigation of mobile robot in indoor environment},
journal = {ISA Transactions},
volume = {109},
pages = {389-400},
year = {2021},
issn = {0019-0578},
doi = {https://doi.org/10.1016/j.isatra.2020.10.023},
url = {https://www.sciencedirect.com/science/article/pii/S0019057820304183},
author = {T. Ran and L. Yuan and J.b. Zhang},
}
