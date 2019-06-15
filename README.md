# Visual-Perception-based-Navigation
## Introduction
**Authors:** T. Ran, L. Yuan, J. Zhang and D. Tang

<img src="https://github.com/rantengsky/Visual-Perception-based-Navigation/blob/master/pictures/fig1.png" width="375">

This project is a real-time indoor navigation system for the mobile robot. It takes the information of an monocular camera as input. The images captured by the camera will be classified by a CNN model. After the classification, the system converts the outputs to specific velocity command through the motion control algorithm (the regular control and the adaptive weighted control). And the robot can navigate by itself under an unknown and dynamic environment.

**Video:** [nav-video](https://pan.baidu.com/s/17zI-3hvoyymZo-VmJHAnJg)   extract code: i0o0
**Model:** [network-model](https://pan.baidu.com/s/1AvrePbG8SOTmA-vhfefkew)   extract code: 6kgs
**Data:** [odom-data](https://pan.baidu.com/s/1JpbxpToTW1kjrBmlFBoqKw)   extract code: qh0q
 

## Prerequisites

### 1. Window 10

### 2. Python 3.X

### 3. TensorFlow-GPU 1.10

### 4. OpenCv 3.4.X

### 5. DataBase: MySQL

To collect encoder data (pose, velocity)

## Contact us

For any issues, please feel free to contact Teng Ran: rantengsky@163.com
