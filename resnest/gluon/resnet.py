##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNets, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring
from __future__ import division

import os
import math
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

from .dropblock import DropBlock
from .splat import SplitAttentionConv

__all__ = ['ResNet', 'Bottleneck']


class Bottleneck(HybridBlock):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, channels, cardinality=1, bottleneck_width=64, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False,
                 dropblock_prob=0, input_size=None, use_splat=False,
                 radix=2, avd=False, avd_first=False, in_channels=None, 
                 split_drop_ratio=0, **kwargs):
        super(Bottleneck, self).__init__()
        group_width = int(channels * (bottleneck_width / 64.)) * cardinality
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.dropblock_prob = dropblock_prob
        self.use_splat = use_splat
        self.avd = avd and (strides > 1 or previous_dilation != dilation)
        self.avd_first = avd_first
        if self.dropblock_prob > 0:
            self.dropblock1 = DropBlock(dropblock_prob, 3, group_width, *input_size)
            if self.avd:
                if avd_first:
                    input_size = _update_input_size(input_size, strides)
                self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
                if not avd_first:
                    input_size = _update_input_size(input_size, strides)
            else:
                input_size = _update_input_size(input_size, strides)
                self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
            self.dropblock3 = DropBlock(dropblock_prob, 3, channels*4, *input_size)
        self.conv1 = nn.Conv2D(channels=group_width, kernel_size=1,
                               use_bias=False, in_channels=in_channels)
        self.bn1 = norm_layer(in_channels=group_width, **norm_kwargs)
        self.relu1 = nn.Activation('relu')
        if self.use_splat:
            self.conv2 = SplitAttentionConv(channels=group_width, kernel_size=3, strides = 1 if self.avd else strides,
                                              padding=dilation, dilation=dilation, groups=cardinality, use_bias=False,
                                              in_channels=group_width, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                              radix=radix, drop_ratio=split_drop_ratio, **kwargs)
        else:
            self.conv2 = nn.Conv2D(channels=group_width, kernel_size=3, strides = 1 if self.avd else strides,
                                   padding=dilation, dilation=dilation, groups=cardinality, use_bias=False,
                                   in_channels=group_width, **kwargs)
            self.bn2 = norm_layer(in_channels=group_width, **norm_kwargs)
            self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=channels*4, kernel_size=1, use_bias=False, in_channels=group_width)
        if not last_gamma:
            self.bn3 = norm_layer(in_channels=channels*4, **norm_kwargs)
        else:
            self.bn3 = norm_layer(in_channels=channels*4, gamma_initializer='zeros',
                                  **norm_kwargs)
        if self.avd:
            self.avd_layer = nn.AvgPool2D(3, strides, padding=1)
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides


class ResNet(HybridBlock):
    """ ResNet Variants Definations
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, cardinality=1, bottleneck_width=64,
                 classes=1000, dilated=False, dilation=1, norm_layer=BatchNorm,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', dropblock_prob=0, input_size=224,
                 use_splat=False, radix=2, avd=False, avd_first=False, split_drop_ratio=0):
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width*2 if deep_stem else 64
        self.radix = radix
        self.split_drop_ratio = split_drop_ratio
        self.avd_first = avd_first
        super(ResNet, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs
        with self.name_scope():
            if not deep_stem:
                self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False, in_channels=3)
            else:
                self.conv1 = nn.HybridSequential(prefix='conv1')
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False, in_channels=3))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False, in_channels=stem_width))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False, in_channels=stem_width))
            input_size = _update_input_size(input_size, 2)
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            input_size = _update_input_size(input_size, 2)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma, use_splat=use_splat,
                                           avd=avd)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma, use_splat=use_splat,
                                           avd=avd)
            input_size = _update_input_size(input_size, 2)
            if dilated or dilation==4:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4, pre_dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            elif dilation==3:
                # special
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2, dilation=2, pre_dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            elif dilation==2:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                input_size = _update_input_size(input_size, 2)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                input_size = _update_input_size(input_size, 2)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)


