##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Ablation Study Models for ResNeSt"""
from .resnet import ResNet, Bottleneck
from mxnet import cpu

__all__ = ['resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d', 'resnest50_fast_4s1x64d',
           'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']








