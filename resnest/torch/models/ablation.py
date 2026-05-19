##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt ablation study models"""

import torch
from .resnet import ResNet, Bottleneck

__all__ = ['resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d', 'resnest50_fast_4s1x64d',
           'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('d8fbf808', 'resnest50_fast_1s1x64d'),
    ('44938639', 'resnest50_fast_2s1x64d'),
    ('f74f3fc3', 'resnest50_fast_4s1x64d'),
    ('32830b84', 'resnest50_fast_1s2x40d'),
    ('9d126481', 'resnest50_fast_2s2x40d'),
    ('41d14ed0', 'resnest50_fast_4s2x40d'),
    ('d4a4f76f', 'resnest50_fast_1s4x24d'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}







