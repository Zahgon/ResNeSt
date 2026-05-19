##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt implemented in Gluon."""

__all__ = ['resnest50', 'resnest101',
           'resnest200', 'resnest269']

from .resnet import ResNet, Bottleneck
from mxnet import cpu




